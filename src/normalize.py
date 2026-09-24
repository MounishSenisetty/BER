"""Text normalisation + light address parsing for business names / addresses.

Produces one enriched frame per record set; every downstream stage (blocking, features)
reads only these precomputed columns, so string work happens once per record, not per pair."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import List

import jellyfish
import pandas as pd

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "co", "corp", "corporation",
    "company", "plc", "gmbh", "pvt", "private", "pllc", "pc", "pa", "sa", "ag", "bv", "nv", "pty",
    "srl", "spa", "sarl", "ab", "oy", "as", "kg", "the", "and",
}

# canonical short forms; applied token-wise to names and addresses alike
ABBREV = {
    # generic business words
    "intl": "international", "int'l": "international", "svc": "services", "svcs": "services",
    "service": "services", "mgmt": "management", "assoc": "associates", "assocs": "associates",
    "bros": "brothers", "ctr": "center", "centre": "center", "mfg": "manufacturing",
    "natl": "national", "grp": "group", "dept": "department", "univ": "university",
    "hosp": "hospital", "restaurants": "restaurant", "rest": "restaurant", "mkt": "market",
    "sys": "systems", "tech": "technologies", "technology": "technologies", "labs": "laboratories",
    "lab": "laboratories", "pharma": "pharmaceuticals", "dr": "dr", "doctor": "dr",
    # street types (USPS-like short forms)
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "place": "pl", "square": "sq", "parkway": "pkwy",
    "highway": "hwy", "hiway": "hwy", "expressway": "expy", "terrace": "ter", "circle": "cir",
    "trail": "trl", "way": "wy", "plaza": "plz", "building": "bldg", "floor": "fl", "suite": "ste",
    "apartment": "apt", "room": "rm", "number": "no", "num": "no", "nr": "no", "mount": "mt",
    "fort": "ft", "saint": "st", "sector": "sec", "nagar": "ngr", "marg": "marg", "main": "main",
    "cross": "crs", "layout": "lyt", "extension": "ext", "industrial": "ind", "estate": "est",
    # directions
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw",
}
UNIT_WORDS = {"ste", "unit", "apt", "fl", "rm", "bldg", "flat", "shop", "office", "no"}

_RE_NONWORD = re.compile(r"[^\w\s]|_", re.UNICODE)
_RE_SPACES = re.compile(r"\s+")
_RE_ALPHA_DIGIT = re.compile(r"(?<=[^\W\d_]{3})(?=\d)|(?<=\d)(?=[^\W\d_]{3})", re.UNICODE)
_RE_NUM = re.compile(r"^\d+[a-z]?$")


def clean(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " and ").replace("'", "").replace("’", "").replace("@", " at ")
    s = _RE_NONWORD.sub(" ", s)
    s = _RE_ALPHA_DIGIT.sub(" ", s)
    return _RE_SPACES.sub(" ", s).strip()


def tokens(s: str) -> List[str]:
    return [ABBREV.get(t, t) for t in clean(s).split()]


@lru_cache(maxsize=1_000_000)
def metaphone(tok: str) -> str:
    try:
        return jellyfish.metaphone(tok) if tok else ""
    except Exception:  # non-latin scripts etc.
        return tok


@lru_cache(maxsize=1_000_000)
def soundex(tok: str) -> str:
    try:
        return jellyfish.soundex(tok) if tok else ""
    except Exception:
        return tok


def _numbers(toks: List[str]) -> List[str]:
    return [t for t in toks if _RE_NUM.match(t)]


def _parse_address(toks: List[str]):
    nums = _numbers(toks)
    postal = next((t for t in reversed(nums) if t.isdigit() and len(t) >= 5), "")
    unit = ""
    for i, t in enumerate(toks[:-1]):
        if t in UNIT_WORDS and _RE_NUM.match(toks[i + 1]):
            unit = toks[i + 1]
            break
    house, street = "", ""
    for i, t in enumerate(toks):
        if _RE_NUM.match(t) and t != postal and t != unit:
            house = t
            nxt = [x for x in toks[i + 1:i + 4] if not _RE_NUM.match(x) and x not in UNIT_WORDS]
            street = nxt[0] if nxt else ""
            break
    return nums, postal, unit, house, street


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised / parsed columns to a canonical (id, name, address, source) frame."""
    out = df.reset_index(drop=True).copy()
    name_toks = [tokens(x) for x in out["name"].tolist()]
    addr_toks = [tokens(x) for x in out["address"].tolist()]

    core_toks = []
    for nt in name_toks:
        ct = [t for t in nt if t not in LEGAL_SUFFIXES]
        core_toks.append(ct if ct else nt)

    out["name_n"] = [" ".join(t) for t in name_toks]
    out["core"] = [" ".join(t) for t in core_toks]
    out["core_toks"] = core_toks
    out["addr_n"] = [" ".join(t) for t in addr_toks]
    out["addr_toks"] = addr_toks
    out["full"] = (out["core"] + " " + out["addr_n"]).str.strip()
    out["compact"] = [ "".join(t) for t in core_toks]
    out["initials"] = ["".join(x[0] for x in t) if len(t) > 1 else "" for t in core_toks]
    out["first_tok"] = [t[0] if t else "" for t in core_toks]
    out["meta_first"] = [metaphone(t) for t in out["first_tok"]]
    out["sdx_first"] = [soundex(t) for t in out["first_tok"]]
    out["phon"] = [" ".join(sorted(metaphone(x) for x in t if not _RE_NUM.match(x))) for t in core_toks]
    out["name_nums"] = [_numbers(t) for t in core_toks]

    parsed = [_parse_address(t) for t in addr_toks]
    out["addr_nums"] = [p[0] for p in parsed]
    out["postal"] = [p[1] for p in parsed]
    out["unit"] = [p[2] for p in parsed]
    out["house"] = [p[3] for p in parsed]
    out["street"] = [p[4] for p in parsed]
    return out
