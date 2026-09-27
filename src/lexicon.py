"""Transliteration lexicon learned from the training ground truth.

About a quarter of India's Source 2 names (and many address parts) are written in an Indic
script, and most of those words are *English* words spelled phonetically: `Gold Software Private
Limited` -> `গোল্ড সফটওয়্যার প্রাইভেট লিমিটেড`. The rule transliterator in normalize.py gets Indian
words right (`राम` -> ram), but turns `সফটওয়্যার` into `saphataoyara`, which shares almost no
character n-grams with `software`, so blocking and every string feature lose the pair.

The training split contains hundreds of thousands of matched (native-script record, Latin Source 1)
pairs, so the mapping can simply be read off them:

1. tokenise both sides; transliterate each native token with the rule table;
2. align the native tokens to the Source 1 tokens with a monotone DP (a native token may also cover
   two Latin tokens, `পশ্চিমবঙ্গ` -> `west bengal`), scoring by Jaro-Winkler of the phonetic forms;
3. count, per native token, which Latin form it aligns to and in how many distinct Source 1
   entities; keep the mapping when it is supported by at least `min_entities` entities with a
   clear majority.

The `min_entities` rule keeps only vocabulary shared across businesses (software, private,
enterprises, healthcare, place names), so a mapping never encodes one entity's own label; that
keeps cross-validation honest. At normalisation time a known native word is replaced by its Latin
form before the rule transliterator runs (normalize.clean).
"""
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rapidfuzz.distance import Indel, JaroWinkler

from .utils import LOG

_RE_INDIC = re.compile("[ऀ-෿]")
_RE_SPLIT = re.compile(r"[\s,.;:()\[\]{}/\\\-&'\"|#@*<>_!?+=]+")
_ZW = dict.fromkeys(map(ord, "‌‍​﻿"))


def native_key(tok: str) -> str:
    """Canonical form of a native-script token (NFC, zero-width joiners removed)."""
    return unicodedata.normalize("NFC", tok).translate(_ZW)


def _latin_tokens(s: str) -> List[str]:
    from .normalize import clean
    return [t for t in clean(s).split() if not t.isdigit()]


def _raw_tokens(s: str) -> List[str]:
    return [t for t in _RE_SPLIT.split(s or "") if t]


def _phon(s: str) -> str:
    from .normalize import _india_phonetic
    return _india_phonetic(s)


def _align(nat: List[str], nat_ph: List[str], lat: List[str], lat_ph: List[str],
           min_sim: float) -> List[Tuple[str, str]]:
    """Monotone alignment of native tokens to Latin tokens (1:1 or 1:2), maximising the summed
    similarity; unmatched tokens on either side are free. Returns (native, latin) pairs."""
    n, m = len(nat), len(lat)
    if n == 0 or m == 0:
        return []
    S = np.zeros((n + 1, m + 1))
    back = np.zeros((n + 1, m + 1), dtype=np.int8)     # 0 skip nat, 1 skip lat, 2 match 1:1, 3 match 1:2
    sim1 = np.array([[JaroWinkler.normalized_similarity(a, b) for b in lat_ph] for a in nat_ph])
    # 1:2 ("pashchimbanga" -> "pashchim banga") scored with the length-sensitive Indel ratio (Jaro-Winkler's
    # prefix bonus would let "templ" swallow "temple street"), and it must beat both single-token options
    def two(a, j):
        s2 = Indel.normalized_similarity(a, lat_ph[j] + lat_ph[j + 1])
        s1 = max(Indel.normalized_similarity(a, lat_ph[j]), Indel.normalized_similarity(a, lat_ph[j + 1]))
        return s2 if s2 >= s1 + 0.1 else 0.0

    sim2 = np.array([[two(a, j) for j in range(m - 1)] for a in nat_ph]) if m > 1 else np.zeros((n, 0))
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, arg = -1.0, 0
            if i > 0 and S[i - 1, j] > best:
                best, arg = S[i - 1, j], 0
            if j > 0 and S[i, j - 1] > best:
                best, arg = S[i, j - 1], 1
            if i > 0 and j > 0 and sim1[i - 1, j - 1] >= min_sim and S[i - 1, j - 1] + sim1[i - 1, j - 1] > best:
                best, arg = S[i - 1, j - 1] + sim1[i - 1, j - 1], 2
            if i > 0 and j > 1 and sim2[i - 1, j - 2] >= min_sim + 0.15 \
                    and S[i - 1, j - 2] + 1.2 * sim2[i - 1, j - 2] > best:
                best, arg = S[i - 1, j - 2] + 1.2 * sim2[i - 1, j - 2], 3
            S[i, j], back[i, j] = best, arg
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        a = back[i, j]
        if a == 0:
            i -= 1
        elif a == 1:
            j -= 1
        elif a == 2:
            out.append((nat[i - 1], lat[j - 1]))
            i, j = i - 1, j - 1
        else:
            out.append((nat[i - 1], lat[j - 2] + " " + lat[j - 1]))
            i, j = i - 1, j - 2
    return out


def _mine(args) -> List[Tuple[str, str, int]]:
    """(native token, latin form, entity index) triples for one block of pairs."""
    from .normalize import _translit_runs
    rec_texts, s1_texts, ent, min_sim = args
    out = []
    for rt, st, e in zip(rec_texts, s1_texts, ent):
        toks = _raw_tokens(rt)
        if not any(_RE_INDIC.search(t) for t in toks):
            continue
        lat = _latin_tokens(st)
        if not lat:
            continue
        nat, nat_ph = [], []
        for t in toks:
            if _RE_INDIC.search(t):
                k = native_key(t)
                ph = _phon("".join(c for c in _translit_runs(k).lower() if c.isalnum()))
                if ph:
                    nat.append(k)
                    nat_ph.append(ph)
        if not nat or len(nat) > 16 or len(lat) > 24:
            continue
        lat_ph = [_phon(x) for x in lat]
        out += [(a, b, e) for a, b in _align(nat, nat_ph, lat, lat_ph, min_sim)]
    return out


def learn_lexicon(pairs_rec: Sequence[str], pairs_s1: Sequence[str], pairs_ent: np.ndarray,
                  min_entities: int = 3, min_share: float = 0.6, min_sim: float = 0.45,
                  n_jobs: int = -1, block: int = 20_000) -> Dict[str, str]:
    """Learn {native token: latin form} from aligned (record text, Source 1 text) pairs."""
    idx = [i for i, t in enumerate(pairs_rec) if isinstance(t, str) and _RE_INDIC.search(t)]
    if not idx:
        return {}
    rec = [pairs_rec[i] for i in idx]
    s1 = [pairs_s1[i] for i in idx]
    ent = np.asarray(pairs_ent)[idx]
    jobs = [(rec[i:i + block], s1[i:i + block], ent[i:i + block], min_sim) for i in range(0, len(rec), block)]
    n_jobs = (os.cpu_count() or 1) if n_jobs is None or n_jobs < 1 else n_jobs
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs))) as ex:
            parts = list(ex.map(_mine, jobs))
    else:
        parts = [_mine(j) for j in jobs]
    ents: Dict[Tuple[str, str], set] = defaultdict(set)
    counts: Dict[str, Counter] = defaultdict(Counter)
    for part in parts:
        for a, b, e in part:
            counts[a][b] += 1
            ents[(a, b)].add(int(e))
    lex = {}
    for a, c in counts.items():
        b, nb = c.most_common(1)[0]
        if nb / sum(c.values()) >= min_share and len(ents[(a, b)]) >= min_entities:
            lex[a] = b
    LOG.info("Transliteration lexicon: %d native tokens mapped (from %d pairs with native script, %d candidates)",
             len(lex), len(idx), len(counts))
    return lex


def learn_from_split(split, n_jobs: int = -1, max_pairs: int = 800_000, seed: int = 0) -> Dict[str, str]:
    """Names and addresses of every matched (record, Source 1) pair with native script in the record."""
    s1_pos = {k: i for i, k in enumerate(split.s1["id"])}
    rec_pos = {k: i for i, k in enumerate(split.rec["id"])}
    ps, pr = [], []
    for s, ms in split.truth.items():
        si = s1_pos.get(s)
        if si is None:
            continue
        for m in ms:
            ri = rec_pos.get(m)
            if ri is not None:
                ps.append(si)
                pr.append(ri)
    if not ps:
        return {}
    ps, pr = np.asarray(ps), np.asarray(pr)
    names = split.rec["name"].to_numpy(dtype=object)
    addrs = split.rec["address"].to_numpy(dtype=object)
    has_nat = np.fromiter(((isinstance(names[r], str) and _RE_INDIC.search(names[r]) is not None)
                           or (isinstance(addrs[r], str) and _RE_INDIC.search(addrs[r]) is not None) for r in pr),
                          dtype=bool, count=len(pr))
    ps, pr = ps[has_nat], pr[has_nat]
    if len(ps) > max_pairs:
        pick = np.random.default_rng(seed).choice(len(ps), size=max_pairs, replace=False)
        ps, pr = ps[pick], pr[pick]
    s1n = split.s1["name"].to_numpy(dtype=object)[ps]
    s1a = split.s1["address"].to_numpy(dtype=object)[ps]
    lex = learn_lexicon(list(names[pr]) + list(addrs[pr]), list(s1n) + list(s1a), np.r_[ps, ps], n_jobs=n_jobs)
    return lex
