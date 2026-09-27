"""Text normalisation + light address parsing for business names / addresses (US, India, France,
and any other country: nothing here filters on the country label, it only selects extra
country-specific abbreviation tables when the label is known).

Pipeline per string:
  1. transliterate Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
     Kannada, Malayalam) to Latin with a rule table -- must run before accent stripping because
     Indic vowel signs are Unicode combining marks
  2. strip web-domain wrappers (www., .com), glue dotted acronyms (S.A.S -> sas, L.L.C. -> llc)
  3. NFKD accent folding, lowercase, & -> and, punctuation -> space, split glued alphanumerics
  4. token-wise abbreviation canonicalisation (common table + per-country overrides)
  5. names: drop legal forms anywhere in the name -> "core" name
     addresses: drop null tokens, canonicalise state names and city aliases, parse house number,
     postal code, unit number and first street token

Every downstream stage reads only these precomputed columns, so string work happens once per
record, never per pair.
"""
from __future__ import annotations

import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd
from rapidfuzz.distance import OSA

try:
    import jellyfish
except ImportError:  # fall back to the built-in phonetic codes below
    jellyfish = None

# ---------------------------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------------------------
LEGAL_SUFFIXES = {
    # US / generic English
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "co", "corp", "corporation",
    "company", "plc", "pllc", "pc", "pa", "dba", "the", "and", "of",
    # India (incl. transliterated Devanagari forms produced by translit())
    "pvt", "private", "public", "opc", "praivet", "privet", "limited", "limitd", "elelpi",
    "kampani", "pablik", "ltda", "p", "td", "lt", "pvtltd", "elaelapi", "elelapi", "limitd", "praivhet",
    # France
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scp", "scop", "selarl", "selas", "sca",
    "gie", "eirl", "ei", "cie", "compagnie", "societe", "et", "de", "du", "des", "la", "le", "les",
    # other common forms
    "gmbh", "ag", "bv", "nv", "pty", "srl", "spa", "ab", "oy", "kg",
}
# long legal words also matched with typos ("Limied", "PIRVATE", "Incorprated"), plus 3-letter
# abbreviations with swapped letters ("Ldt", "Ptv") -- otherwise a mistyped legal form stays in
# the core name and looks like the unexplained extra token of a branch distractor
_LEGAL_FUZZY = ("limited", "private", "company", "corporation", "incorporated", "compagnie", "societe",
                "partnership")
_LEGAL_ANAGRAM = {"".join(sorted(w)): w for w in ("ltd", "pvt")}


# legal forms as they appear with broken spacing / typos at either edge of a name:
# "PvtL td", "PVTLTD", "Pvtc Ltd", "LcLP", "OCRP", "PvtLtd Gupta Jewellers"
_LEGAL_GLUED = ("pvtltd", "privatelimited", "pvtlimited", "privateltd", "coltd", "ltd", "pvt", "llp",
                "llc", "inc", "corp", "limited", "private", "incorporated", "corporation", "company", "plc")


# French forms mistyped at a name edge ("sSCI", "TSAS", "EUvRL", "S..S" -> "s s"); only for
# France-labelled records, since "seas" / "sync" / "earl" are one edit from sas / snc / eurl
_LEGAL_GLUED_FR = ("sas", "sasu", "sarl", "eurl", "sci", "snc", "selarl", "scop")


@lru_cache(maxsize=1_000_000)
def is_glued_legal(s: str, joined: bool = False, fr: bool = False, letters: bool = False) -> bool:
    """`joined`: s spans several tokens -- one edit at most, so a real neighbouring word is never
    absorbed ("privatelimited" + "om" is two edits away from "privatelimited").
    `letters`: s is a run of single-letter tokens ("s s" from a mangled "S.A.S")."""
    if s in _LEGAL_GLUED or (fr and s in _LEGAL_GLUED_FR):
        return True
    words = _LEGAL_GLUED + (_LEGAL_GLUED_FR if fr else ())
    if len(s) < 4 and not (fr and letters and len(s) >= 2):   # "inn", "ink" must not become "inc"
        return False
    return any(OSA.distance(s, w, score_cutoff=2) <= (2 if len(w) >= 8 and not joined else 1)
               for w in words if abs(len(w) - len(s)) <= 2)


def _peel_legal_edges(toks: List[str], legal: List[bool], max_join: int = 4, fr: bool = False) -> List[int]:
    """Indices of `toks` left after repeatedly removing legal forms (possibly split / glued /
    mistyped over up to `max_join` tokens) from the end and the start of the name."""
    idx = list(range(len(toks)))
    for from_end in (True, False):
        changed = True
        while changed and len(idx) > 1:
            changed = False
            for k in range(min(max_join, len(idx) - 1), 0, -1):
                part = idx[-k:] if from_end else idx[:k]
                if all(legal[i] for i in part) or (
                        not any(toks[i].isdigit() for i in part)     # branch numbers stay ("II Inc")
                        and is_glued_legal("".join(toks[i] for i in part), k > 1, fr,
                                           k > 1 and all(len(toks[i]) == 1 for i in part))):
                    idx = idx[:-k] if from_end else idx[k:]
                    changed = True
                    break
    return idx


# canonical legal form of a name ("Pvt. Ltd." / "Private (Limited)" / "PvtL td" -> pvtltd): two S1
# entities with the same core name are often told apart only by it ("... LLC" vs "... Inc")
_LEGAL_CANON = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co",
    "company": "co", "kampani": "co", "ltd": "ltd", "limited": "ltd", "limitd": "ltd", "coltd": "ltd",
    "pvt": "pvtltd", "pvtltd": "pvtltd", "privatelimited": "pvtltd", "pvtlimited": "pvtltd",
    "privateltd": "pvtltd", "praivetlimitd": "pvtltd", "privetlimitd": "pvtltd", "llc": "llc", "llp": "llp",
    "elelpi": "llp", "elaelapi": "llp", "elelapi": "llp", "plc": "plc", "pllc": "pllc", "sarl": "sarl",
    "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci", "snc": "snc", "gmbh": "gmbh",
}
_LEGAL_STOP = {"the", "and", "of", "et", "de", "du", "des", "la", "le", "les", "dba"}


@lru_cache(maxsize=1_000_000)
def canon_legal(s: str) -> str:
    """Joined removed-legal tokens -> canonical form; '' when absent or unrecognisable."""
    if not s or s in _LEGAL_CANON:
        return _LEGAL_CANON.get(s, "")
    if len(s) < 4:
        return ""
    best, dist = "", 3
    for k, v in _LEGAL_CANON.items():
        if abs(len(k) - len(s)) <= 2 and len(k) >= 3:
            d = OSA.distance(s, k, score_cutoff=2)
            if d < dist and d <= (2 if len(k) >= 8 else 1):
                best, dist = v, d
    return best


WEB_TOKENS = {"www", "http", "https"}
HONORIFICS = {"smt", "shrimati", "mr", "mrs", "kumari", "sh"}
_NOT_LEGAL = {"privacy", "privilege", "privileged", "privy", "limitless", "corporate", "companion", "companions"}


@lru_cache(maxsize=1_000_000)
def is_legal(t: str) -> bool:
    """Legal-form token, tolerating the misspellings seen in the data (Privhea, Liimted, Limitend,
    Incorprated) and 3-letter abbreviations with swapped letters (Ldt, Ptv)."""
    if t in LEGAL_SUFFIXES:
        return True
    if len(t) == 3:
        return "".join(sorted(t)) in _LEGAL_ANAGRAM
    if len(t) < 6 or not t.isalpha() or t in _NOT_LEGAL:
        return False
    if t.startswith(("priv", "praiv", "pirai", "limit", "lmit", "incorp", "corpor")):
        return True
    return any(OSA.distance(t, w, score_cutoff=1 if len(w) < 9 else 2) <= (1 if len(w) < 9 else 2)
               for w in _LEGAL_FUZZY if abs(len(w) - len(t)) <= 2)


_RE_LEET = re.compile(r"(?<=[A-Za-z])[01](?=[A-Za-z])|\b[0156](?=[A-Za-z]{2,})(?!(?:st|nd|rd|th)\b)")
_LEET = {"0": "o", "1": "l", "5": "s", "6": "g"}


def _unleet(m) -> str:
    return _LEET[m.group(0)]


NULL_TOKENS = {"null", "none", "nan", "nil", "undefined"}

ABBREV_COMMON = {
    # business words
    "intl": "international", "svc": "services", "svcs": "services", "service": "services",
    "mgmt": "management", "assoc": "associates", "assocs": "associates", "bros": "brothers",
    "ctr": "center", "centre": "center", "mfg": "manufacturing", "natl": "national", "grp": "group",
    "dept": "department", "univ": "university", "hosp": "hospital", "mkt": "market",
    "sys": "systems", "tech": "technologies", "technology": "technologies", "labs": "laboratories",
    "lab": "laboratories", "pharma": "pharmaceuticals", "ent": "enterprises",
    "enterprise": "enterprises", "entp": "enterprises", "engg": "engineering", "eng": "engineering",
    "sri": "shri", "shree": "shri", "shre": "shri", "sree": "shri", "doctor": "dr",
    # branch / phase numerals -> digits, so "Store II" == "Store 2" and the India phonetic
    # repeat-collapse cannot shrink "ii" into a one-letter typo
    "ii": "2", "iii": "3", "iv": "4",
    # address words (short canonical forms)
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd", "boulevard": "blvd",
    "blv": "blvd", "bd": "blvd", "drive": "dr", "lane": "ln", "court": "ct", "place": "pl",
    "square": "sq", "parkway": "pkwy", "highway": "hwy", "expressway": "expy", "terrace": "ter",
    "circle": "cir", "trail": "trl", "plaza": "plz", "building": "bldg", "floor": "fl",
    "suite": "ste", "apartment": "apt", "room": "rm", "number": "no", "num": "no",
    "mount": "mt", "fort": "ft",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw",
    # ordinal words -> digit ordinals ("531 FIFTEENTH AVE" == "531 15th Avenue")
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
    "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th", "eleventh": "11th",
    "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
    "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th", "nineteenth": "19th",
    "twentieth": "20th", "thirtieth": "30th", "fortieth": "40th", "fiftieth": "50th",
}
ABBREV_COUNTRY = {
    "india": {
        "nr": "near", "end": "and", "opp": "opposite", "oppo": "opposite", "bhd": "behind", "nagar": "ngr",
        "sector": "sec", "sect": "sec", "phase": "ph", "block": "blk", "extension": "extn",
        "ext": "extn", "dist": "district", "distt": "district", "tq": "taluk", "tal": "taluk",
        "taluka": "taluk", "hno": "no", "industrial": "ind", "indl": "ind", "estate": "est",
        "cross": "crs", "layout": "lyt", "colony": "col", "chowk": "chk", "marg": "marg",
        "bengaluru": "bangalore", "mumbai": "bombay", "gurugram": "gurgaon", "kolkata": "calcutta",
        "chennai": "madras", "puducherry": "pondicherry", "thiruvananthapuram": "trivandrum",
        "vadodara": "baroda", "prayagraj": "allahabad", "mysuru": "mysore", "kochi": "cochin",
        "belagavi": "belgaum", "mangaluru": "mangalore", "shimla": "simla", "pune": "poona",
    },
    "france": {
        "r": "rue", "st": "saint", "ste": "sainte", "bd": "boulevard", "blvd": "boulevard",
        "bld": "boulevard", "boul": "boulevard", "av": "avenue", "ave": "avenue", "imp": "impasse",
        "ch": "chemin", "chem": "chemin", "pl": "place", "rte": "route", "sq": "square",
        "fg": "faubourg", "fbg": "faubourg", "all": "allee", "qu": "quai", "crs": "cours",
        "res": "residence", "resid": "residence", "zi": "zone", "za": "zone", "zac": "zone",
        "cedex": "", "bp": "",
    },
}
UNIT_WORDS = {"ste", "unit", "apt", "fl", "rm", "bldg", "flat", "shop", "office"}
# canonical street types (after abbreviation mapping), particles / directionals skipped when
# looking for the street name, and French house-number suffixes
STREET_TYPES = {"st", "ave", "rd", "blvd", "dr", "ln", "ct", "pl", "sq", "pkwy", "hwy", "expy", "ter",
                "cir", "trl", "plz", "way", "rue", "chemin", "impasse", "route", "allee", "quai", "cours",
                "faubourg", "passage", "residence", "marg"}
STREET_PARTICLES = {"de", "du", "des", "la", "le", "les", "l", "d", "the", "of", "n", "s", "e", "w",
                    "ne", "nw", "se", "sw"}
HOUSE_SUFFIXES = {"bis", "ter", "quater"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "andaman and nicobar islands": "an", "chandigarh": "ch",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "delhi": "dl", "new delhi": "dl",
    "nct of delhi": "dl", "dilli": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "ladakh": "la", "lakshadweep": "ld", "puducherry": "py", "pondicherry": "py",
    # common transliteration outputs of native-script state names
    "maharashtr": "mh", "karnatak": "ka", "gujrat": "gj", "rajasthaan": "rj",
    "uttar pradesh": "up", "madhy pradesh": "mp", "tamil naadu": "tn", "tamizh naadu": "tn",
    "pashchim bangal": "wb", "haryaana": "hr", "bihaar": "br", "telangaan": "ts",
}
_STATE_TABLES = {"us": US_STATES, "usa": US_STATES, "united states": US_STATES, "india": IN_STATES}


def _phrase_regex(table: Dict[str, str]):
    keys = sorted(table, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")


_STATE_RE = {k: (_phrase_regex(v), v) for k, v in _STATE_TABLES.items()}

# ---------------------------------------------------------------------------------------------
# Indic transliteration (rule table on the ISCII-aligned Unicode blocks)
# ---------------------------------------------------------------------------------------------
_INDIC_START, _INDIC_END = 0x0900, 0x0DFF
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
         0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
         0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph",
         0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l",
         0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
         0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_VOWELS = {0x04: "a", 0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu",
           0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o",
           0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
_MATRAS = {0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "ri",
           0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
           0x62: "li", 0x63: "li", 0x57: "au"}
_VIRAMA, _SIGNS = 0x4D, {0x01: "n", 0x02: "n", 0x03: "h", 0x00: "n"}
_DROP_FINAL_SCHWA = {0x0900, 0x0A00, 0x0A80}  # Devanagari, Gurmukhi, Gujarati


def _is_indic(c: str) -> bool:
    return _INDIC_START <= ord(c) <= _INDIC_END


def translit(s: str) -> str:
    """Rule-based transliteration of Indic scripts to Latin. Non-Indic characters pass through."""
    out: List[str] = []
    n = len(s)
    i = 0
    while i < n:
        c = s[i]
        o = ord(c)
        if not (_INDIC_START <= o <= _INDIC_END):
            out.append(c)
            i += 1
            continue
        block = o & ~0x7F
        off = o - block
        if off in _CONS:
            out.append(_CONS[off])
            j = i + 1
            if j < n and ord(s[j]) - block == 0x3C:  # nukta
                j += 1
            nxt = ord(s[j]) - block if j < n and _is_indic(s[j]) else None
            if nxt in _MATRAS:
                out.append(_MATRAS[nxt])
                j += 1
            elif nxt == _VIRAMA:
                j += 1
            elif nxt is None or nxt in _SIGNS:
                # Hindi-style final schwa deletion, except after a conjunct (aditya, maharashtra)
                after_conjunct = (i > 0 and _is_indic(s[i - 1]) and ord(s[i - 1]) - block == _VIRAMA
                                  and off in (0x2F, 0x30, 0x35))  # ...ya / ...ra / ...va
                if not (nxt is None and block in _DROP_FINAL_SCHWA and not after_conjunct):
                    out.append("a")
            else:
                out.append("a")
            i = j
            continue
        if off in _VOWELS:
            out.append(_VOWELS[off])
        elif off in _SIGNS:
            out.append(_SIGNS[off])
        elif 0x66 <= off <= 0x6F:
            out.append(str(off - 0x66))
        elif off in (0x64, 0x65):  # danda
            out.append(" ")
        i += 1
    return "".join(out)


_RE_INDIC = re.compile("[\u0900-\u0DFF]")
_RE_INDIC_RUN = re.compile("[ऀ-෿]+")
_RE_DOUBLE_VOWEL = re.compile(r"([aeiou])\1+")


def _translit_runs(s: str) -> str:
    """Transliterate only the Indic runs; doubled vowels are collapsed inside those runs only."""
    return _RE_INDIC_RUN.sub(lambda m: _RE_DOUBLE_VOWEL.sub(r"\1", translit(m.group(0))), s)

# ---------------------------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------------------------
_RE_URL = re.compile(r"(?:https?://)?(?:www\.)?([\w-]+)\.(?:com|net|org|biz|info|co\.in|in|co|fr|us|io)\b",
                     re.IGNORECASE)
_RE_DOTTED = re.compile(r"\b(?:[^\W\d_]\.){2,}(?:[^\W\d_]\b)?", re.UNICODE)
_RE_NONWORD = re.compile(r"[^\w\s]|_", re.UNICODE)
_RE_SPACES = re.compile(r"\s+")
_RE_ALPHA_DIGIT = re.compile(r"(?<=[^\W\d_]{3})(?=\d)|(?<=\d)(?=[^\W\d_]{3})", re.UNICODE)
_RE_NUM = re.compile(r"^\d+[a-z]?$")


# native-script word -> Latin form, learned from the training ground truth (src/lexicon.py);
# set once per process (train / inference and every normalisation worker) before prepare()
_LEXICON: Dict[str, str] = {}
_RE_INDIC_WORD = re.compile("[\u0900-\u0DFF\u200b-\u200d]+")


def set_lexicon(lex) -> None:
    global _LEXICON
    _LEXICON = dict(lex or {})


def get_lexicon() -> Dict[str, str]:
    return _LEXICON


def _lex_sub(m) -> str:
    from .lexicon import native_key
    v = _LEXICON.get(native_key(m.group(0)))
    return f" {v} " if v else m.group(0)


def clean(s: str) -> str:
    """Script/format normalisation shared by names and addresses. Returns lowercase ASCII-ish text."""
    if not isinstance(s, str) or not s:
        return ""
    if _RE_INDIC.search(s):
        if _LEXICON:
            s = _RE_INDIC_WORD.sub(_lex_sub, s)
        s = _translit_runs(s)
    s = _RE_URL.sub(r"\1", s)
    s = _RE_DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " and ").replace("'", "").replace("’", "").replace("@", " at ")
    s = _RE_NONWORD.sub(" ", s)
    s = _RE_ALPHA_DIGIT.sub(" ", s)
    return _RE_SPACES.sub(" ", s).strip()


def country_key(country: str) -> str:
    return (country or "").strip().lower()


_RE_LEAD_ZEROS = re.compile(r"^0+(?=\d)")


def tokens(s: str, country: str = "") -> List[str]:
    table = ABBREV_COUNTRY.get(country_key(country))
    out = []
    for t in clean(s).split():
        if t in NULL_TOKENS:
            continue
        if t[0] == "0":
            t = _RE_LEAD_ZEROS.sub("", t)          # "002839" -> "2839", "00540k" -> "540k"
        if table is not None:                   # country table first, then the shared canonical
            t = table.get(t, t)                 # forms: FR "pl." -> place -> pl == "Place" -> pl
        t = ABBREV_COMMON.get(t, t)
        if t:
            out.append(t)
    return out


# French mail-routing tokens carry no location: "CEDEX 08", "BP 1234", "CS 30012"
_RE_FR_ROUTING = re.compile(r"\b(?:cedex|bp|cs)(?: \d{1,5})?\b")


def address_tokens(s: str, country: str = "") -> List[str]:
    c = clean(s)
    if country_key(country) == "france":
        c = _RE_FR_ROUTING.sub(" ", c)
    st = _STATE_RE.get(country_key(country))
    if st is not None:
        rx, table = st
        c = rx.sub(lambda m: table[m.group(1)], c)
    toks = [t for t in tokens(c, country) if t not in WEB_TOKENS]
    # drop "PO BOX 4442" / "P O BOX 12": mailing boxes are noise added to street addresses
    out, i = [], 0
    while i < len(toks):
        if toks[i] == "box" and out and out[-1] == "po" or (toks[i] == "box" and out[-2:] == ["p", "o"]):
            out = out[:-1] if out[-1] == "po" else out[:-2]
            i += 2 if i + 1 < len(toks) and toks[i + 1].isdigit() else 1
            continue
        out.append(toks[i])
        i += 1
    return out


# ---------------------------------------------------------------------------------------------
# Phonetics
# ---------------------------------------------------------------------------------------------
_SDX = {c: d for d, cs in {"1": "bfpv", "2": "cgjkqsxz", "3": "dt", "4": "l", "5": "mn", "6": "r"}.items()
        for c in cs}
_RE_VOWELS = re.compile(r"(?<!^)[aeiouyhw]")
_RE_REPEAT = re.compile(r"(.)\1+")


def _soundex(tok: str) -> str:
    tok = "".join(c for c in tok if "a" <= c <= "z")
    if not tok:
        return ""
    out, prev = [tok[0].upper()], _SDX.get(tok[0], "")
    for c in tok[1:]:
        d = _SDX.get(c, "")
        if d and d != prev:
            out.append(d)
        if c not in "hw":
            prev = d
    return ("".join(out) + "000")[:4]


def _skeleton(tok: str) -> str:
    """Crude metaphone stand-in: common digraphs, drop non-leading vowels, squeeze repeats."""
    for a, b in (("ph", "f"), ("ck", "k"), ("sch", "sk"), ("sh", "x"), ("ch", "x"), ("th", "0"),
                 ("gh", ""), ("kn", "n"), ("wr", "r"), ("q", "k"), ("z", "s"), ("c", "k"), ("v", "f")):
        tok = tok.replace(a, b)
    return _RE_REPEAT.sub(r"\1", _RE_VOWELS.sub("", tok)).upper()


@lru_cache(maxsize=2_000_000)
def metaphone(tok: str) -> str:
    if not tok:
        return ""
    try:
        return jellyfish.metaphone(tok) if jellyfish else _skeleton(tok)
    except Exception:
        return tok


@lru_cache(maxsize=2_000_000)
def soundex(tok: str) -> str:
    if not tok:
        return ""
    try:
        return jellyfish.soundex(tok) if jellyfish else (_soundex(tok) or tok)
    except Exception:
        return tok


# ---------------------------------------------------------------------------------------------
# Record preparation
# ---------------------------------------------------------------------------------------------
def _until_number(toks: List[str], start: int, n: int) -> List[str]:
    """Up to n tokens from `start`, stopping at the next number (house suffixes are skipped)."""
    out = []
    for x in toks[start:start + n]:
        if _RE_NUM.match(x):
            break
        out.append(x)
    return out


def _parse_address(toks: List[str]):
    nums = [t for t in toks if _RE_NUM.match(t)]
    # a number followed by a street type is the house number, whatever the component order:
    # in "md 21251 dundalk 10290 fawn ave" the postal code is 21251, not the last 5-digit number
    street_num = next((i for i, t in enumerate(toks) if _RE_NUM.match(t) and any(
        x in STREET_TYPES for x in _until_number(toks, i + 1, 5))), None)
    skip = toks[street_num] if street_num is not None else None
    postal = next((t for t in reversed(nums) if t.isdigit() and 5 <= len(t) <= 6 and t != skip), "")
    unit = ""
    for i in range(len(toks) - 1):
        if toks[i] in UNIT_WORDS and _RE_NUM.match(toks[i + 1]):
            unit = toks[i + 1]
            break
    house, street, stype = "", "", ""
    for i, t in enumerate(toks):
        if (i == street_num) if street_num is not None else (_RE_NUM.match(t) and t != postal and t != unit):
            house = t
            j = i + 1
            if j < len(toks) and toks[j] in HOUSE_SUFFIXES:     # "20 bis Bd ..." is house 20
                j += 1
            # street NAME, not its type: "rue de dieppe" -> dieppe, "w 5th ave" -> 5th
            for x in toks[j:j + 5]:
                if _RE_NUM.match(x) or x in UNIT_WORDS:
                    continue
                if x in STREET_TYPES:
                    stype = stype or x
                elif x not in STREET_PARTICLES and not street:
                    street = x
                if street and stype:
                    break
            break
    return nums, postal, unit, house, street, stype


COLUMNS = ["name_n", "core", "core_toks", "addr_n", "addr_toks", "full", "compact", "initials",
           "first_tok", "meta_first", "sdx_first", "phon", "name_nums", "addr_nums", "postal",
           "unit", "house", "street", "translit", "legal", "street_type"]


_RE_REPEAT_CHAR = re.compile(r"([a-z])\1+")
_INDIA_PHON = (("ph", "f"), ("oo", "u"), ("ee", "i"), ("x", "ksh"), ("w", "v"))


@lru_cache(maxsize=2_000_000)
def _india_phonetic(tok: str) -> str:
    """Bring Latin spellings of Indian names and transliterated Devanagari together
    (phuds / foods -> fuds, shree -> shri, laxmi -> lakshmi, jewellers -> jevelers)."""
    for a, b in _INDIA_PHON:
        tok = tok.replace(a, b)
    return _RE_REPEAT_CHAR.sub(r"\1", tok)


def _prep_one(name: str, address: str, country: str):
    ckey = country_key(country)
    nt = [t for t in tokens(_RE_LEET.sub(_unleet, name or ""), ckey) if t not in WEB_TOKENS]
    if len(nt) > 2 and nt[0] == "m" and nt[1] == "s":            # "M/s Foo Traders"
        nt = nt[2:]
    drop = HONORIFICS | ({"pra", "li"} if ckey == "india" else set())    # India: "प्रा. लि." = Pvt. Ltd.
    legal = [is_legal(t) or t in drop for t in nt]      # before phonetics mangle typos ("lximited")
    keep = _peel_legal_edges(nt, legal, fr=ckey == "france")
    raw = nt
    if ckey == "india":
        nt = [_india_phonetic(t) for t in nt]
    kept = [i for i in keep if not (legal[i] or is_legal(nt[i]))]
    ct = [nt[i] for i in kept] or nt
    kept_set = set(kept)
    lform = canon_legal("".join(raw[i] for i in range(len(raw)) if i not in kept_set
                                and raw[i] not in _LEGAL_STOP and raw[i] not in drop)) if kept else ""
    at = address_tokens(address, ckey)
    core = " ".join(ct)
    addr = " ".join(at)
    first = ct[0] if ct else ""
    nums, postal, unit, house, street, stype = _parse_address(at)
    return (" ".join(nt), core, ct, addr, at, (core + " " + addr).strip(), "".join(ct),
            "".join(x[0] for x in ct) if len(ct) > 1 else "", first, metaphone(first), soundex(first),
            " ".join(sorted(metaphone(x) for x in ct if not _RE_NUM.match(x))),
            [t for t in ct if _RE_NUM.match(t)], nums, postal, unit, house, street,
            bool(_RE_INDIC.search(name or "")), lform, stype)


def _prep_block(args):
    names, addrs, countries = args
    rows = [_prep_one(n, a, c) for n, a, c in zip(names, addrs, countries)]
    return pd.DataFrame(rows, columns=COLUMNS)


def prepare(df: pd.DataFrame, n_jobs: int = -1, block: int = 50_000) -> pd.DataFrame:
    """Add normalised / parsed columns to a canonical (id, name, address, country, source) frame."""
    out = df.reset_index(drop=True)
    names = out["name"].tolist()
    addrs = out["address"].tolist()
    ctry = out["country"].tolist() if "country" in out.columns else [""] * len(out)
    jobs = [(names[i:i + block], addrs[i:i + block], ctry[i:i + block]) for i in range(0, len(out), block)]
    n_jobs = (os.cpu_count() or 1) if n_jobs is None or n_jobs < 1 else n_jobs
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs)), initializer=set_lexicon,
                                 initargs=(_LEXICON,)) as ex:
            parts = list(ex.map(_prep_block, jobs))
    else:
        parts = [_prep_block(j) for j in jobs]
    prep = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    prep.index = out.index
    return pd.concat([out, prep], axis=1)
