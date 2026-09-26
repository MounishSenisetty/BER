"""Pairwise feature extraction for one chunk of candidate pairs (all pairs of a record chunk).

Everything is vectorised over pairs:
  * TF-IDF cosines         -> sparse row-wise products (scipy); dense SVD cosines from blocking
  * token-set overlaps     -> idf-weighted incidence matrices built for the chunk, idf taken from
                              the country's Source 1 so values are consistent across chunks
  * fuzzy string metrics   -> rapidfuzz.process.cpdist (C++, multi-threaded, element-wise)
  * exact / phonetic flags -> numpy comparisons of per-record precomputed keys
  * context features       -> rank / margin of the pair among the *record's* candidates
                              (a record's candidates always live in the same chunk)
No country label is used as a feature, so the model transfers to unseen countries (France).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, LCSseq

from .blocking import BLOCKERS, CountryIndex, RecordMats, rowwise_dot


def _incidence(l1: Sequence[List[str]], l2: Sequence[List[str]]):
    vocab: Dict[str, int] = {}

    def build(lists):
        indptr, indices = [0], []
        for toks in lists:
            ids = {vocab.setdefault(t, len(vocab)) for t in toks}
            indices.extend(ids)
            indptr.append(len(indices))
        return indptr, indices

    p1, i1 = build(l1)
    p2, i2 = build(l2)
    V = max(len(vocab), 1)
    B1 = sp.csr_matrix((np.ones(len(i1), np.float32), i1, p1), shape=(len(l1), V))
    B2 = sp.csr_matrix((np.ones(len(i2), np.float32), i2, p2), shape=(len(l2), V))
    return B1, B2, list(vocab)


def _overlap_feats(prefix: str, l1u, l2u, ja, jb, idf: Dict[str, float] = None, default_idf: float = 1.0):
    """Overlap features for pairs (l1u[ja[k]], l2u[jb[k]]) -- l1u/l2u are the chunk's unique rows."""
    B1, B2, vocab = _incidence(l1u, l2u)
    cnt = rowwise_dot(B1, B2, ja, jb)
    n1 = np.asarray(B1.sum(1)).ravel()[ja]
    n2 = np.asarray(B2.sum(1)).ravel()[jb]
    both_empty = (n1 == 0) & (n2 == 0)
    out = {f"{prefix}_common": cnt,
           f"{prefix}_only1": np.where(both_empty, np.nan, n1 - cnt),
           f"{prefix}_only2": np.where(both_empty, np.nan, n2 - cnt)}
    if idf is not None:
        w = np.fromiter((idf.get(t, default_idf) for t in vocab), dtype=np.float32, count=len(vocab)) \
            if vocab else np.ones(1, np.float32)
        W1 = B1.multiply(w).tocsr()
        W2 = B2.multiply(w).tocsr()
        inter = rowwise_dot(W1, B2, ja, jb)
        w1 = np.asarray(W1.sum(1)).ravel()[ja]
        w2 = np.asarray(W2.sum(1)).ravel()[jb]
        union = w1 + w2 - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{prefix}_wjac"] = np.where(union > 0, inter / union, np.nan)
            out[f"{prefix}_wcont1"] = np.where(w1 > 0, inter / w1, np.nan)
            out[f"{prefix}_wcont2"] = np.where(w2 > 0, inter / w2, np.nan)
        out[f"{prefix}_winter"] = inter
        out[f"{prefix}_wonly1"] = w1 - inter          # unexplained rare mass ("Annex", "Outlet")
        out[f"{prefix}_wonly2"] = w2 - inter
        out[f"{prefix}_wmaxonly"] = np.maximum(w1, w2) - inter
    return out


def _cp(a, b, scorer, n_jobs: int) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=n_jobs, dtype=np.float32)


def _eq_nonempty(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    miss = (a == "") | (b == "")
    return np.where(miss, np.nan, (a == b).astype(np.float32))


def _group_context(g: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rank (1 = best) of x inside its group and margin to the best *other* member (vectorised)."""
    n = len(x)
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    order = np.lexsort((-x, g))
    gs, xs = g[order], x[order]
    start = np.r_[True, gs[1:] != gs[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    size = np.diff(np.r_[first, n])
    pos = np.arange(n) - first[gid]
    top1 = xs[first][gid]
    second = np.where(size[gid] > 1, xs[np.minimum(first + 1, n - 1)][gid], np.nan)
    gap_sorted = xs - np.where(pos == 0, second, top1)
    rank = np.empty(n, np.float32)
    gap = np.empty(n, np.float32)
    rank[order] = pos + 1
    gap[order] = gap_sorted
    return rank, gap


def _objarr(values) -> np.ndarray:
    return np.asarray(list(values) + [None], dtype=object)[:-1]


def compute_features(idx: CountryIndex, rc: pd.DataFrame, cand: pd.DataFrame, mats: RecordMats,
                     n_jobs: int = -1) -> pd.DataFrame:
    ia = cand["s1"].to_numpy()
    ib = cand["rec"].to_numpy()
    F: Dict[str, np.ndarray] = {}

    def s1col(c):
        return idx.col(c)[ia]

    rcols: Dict[str, np.ndarray] = {}

    def reccol(c):
        if c not in rcols:
            rcols[c] = _objarr(rc[c].tolist())
        return rcols[c][ib]

    # ---- blocking provenance / dense cosines ----------------------------------------------
    for c in BLOCKERS + ["n_blockers", "knn_name_rank", "knn_full_rank", "key_rank", "cheap", "cheap_rank",
                         "cos_name_d", "cos_full_d", "key_score"]:
        F[c] = cand[c].to_numpy()
    F["rec_source"] = rc["source"].to_numpy()[ib].astype(np.int8)
    F["translit1"] = idx.s1["translit"].to_numpy()[ia].astype(np.float32)
    F["translit2"] = rc["translit"].to_numpy()[ib].astype(np.float32)

    # ---- exact sparse tf-idf cosines ------------------------------------------------------
    F["cos_name_c"] = rowwise_dot(idx.name_X, mats.name_X, ia, ib)
    F["cos_full_c"] = rowwise_dot(idx.full_X, mats.full_X, ia, ib)
    F["cos_addr_c"] = rowwise_dot(idx.addr_X, mats.addr_X, ia, ib)

    # ---- token-set overlaps (chunk-local incidence, global idf) ---------------------------
    u1, ja = np.unique(ia, return_inverse=True)
    u2, jb = np.unique(ib, return_inverse=True)
    n_tok = max(idx.N, 1)
    dflt = math.log(n_tok + 1) + 1
    core1 = idx.col("core_toks")[u1]
    core2 = _objarr(rc["core_toks"].tolist())[u2]
    addr1 = idx.col("addr_toks")[u1]
    addr2 = _objarr(rc["addr_toks"].tolist())[u2]
    F.update(_overlap_feats("tok_name", core1, core2, ja, jb, idx.tok_idf, dflt))
    F.update(_overlap_feats("tok_addr", addr1, addr2, ja, jb, idx.addr_idf, dflt))
    full_idf = {**idx.addr_idf, **idx.tok_idf}
    F.update(_overlap_feats("tok_full", [a + b for a, b in zip(core1, addr1)],
                            [a + b for a, b in zip(core2, addr2)], ja, jb, full_idf, dflt))
    F.update(_overlap_feats("num_addr", idx.col("addr_nums")[u1], _objarr(rc["addr_nums"].tolist())[u2], ja, jb))
    F.update(_overlap_feats("num_name", idx.col("name_nums")[u1], _objarr(rc["name_nums"].tolist())[u2], ja, jb))

    # ---- fuzzy string metrics -------------------------------------------------------------
    c1, c2 = s1col("core"), reccol("core")
    a1, a2 = s1col("addr_n"), reccol("addr_n")
    n_empty = (c1 == "") | (c2 == "")
    a_empty = (a1 == "") | (a2 == "")
    for nm, sc in [("name_ratio", fuzz.ratio), ("name_pratio", fuzz.partial_ratio),
                   ("name_tsort", fuzz.token_sort_ratio), ("name_tset", fuzz.token_set_ratio),
                   ("name_jw", JaroWinkler.normalized_similarity)]:
        F[nm] = np.where(n_empty, np.nan, _cp(c1, c2, sc, n_jobs))
    F["name_full_ratio"] = _cp(s1col("name_n"), reccol("name_n"), fuzz.ratio, n_jobs)
    for nm, sc in [("addr_ratio", fuzz.ratio), ("addr_pratio", fuzz.partial_ratio),
                   ("addr_tset", fuzz.token_set_ratio), ("addr_tsort", fuzz.token_sort_ratio)]:
        F[nm] = np.where(a_empty, np.nan, _cp(a1, a2, sc, n_jobs))
    F["full_tset"] = _cp(s1col("full"), reccol("full"), fuzz.token_set_ratio, n_jobs)
    F["phon_ratio"] = _cp(s1col("phon"), reccol("phon"), fuzz.ratio, n_jobs)
    comp1, comp2 = s1col("compact"), reccol("compact")
    F["compact_jw"] = _cp(comp1, comp2, JaroWinkler.normalized_similarity, n_jobs)
    # characters on each side not explained by the longest common subsequence: a typo leaves ~1,
    # a glued branch suffix ("IndustriesII", "burgerii", "Annex") leaves several on one side only
    lcs = _cp(comp1, comp2, LCSseq.similarity, n_jobs)
    cl1 = np.fromiter((len(x) for x in comp1), np.float32, len(comp1))
    cl2 = np.fromiter((len(x) for x in comp2), np.float32, len(comp2))
    F["compact_extra1"], F["compact_extra2"] = cl1 - lcs, cl2 - lcs
    F["compact_extra_max"] = np.maximum(cl1, cl2) - lcs
    F["first_jw"] = _cp(s1col("first_tok"), reccol("first_tok"), JaroWinkler.normalized_similarity, n_jobs)
    F["name2_in_addr1"] = np.where(n_empty | (a1 == ""), np.nan, _cp(c2, a1, fuzz.partial_ratio, n_jobs))

    # ---- exact / phonetic / parsed-address flags ------------------------------------------
    F["core_eq"] = (c1 == c2).astype(np.float32)
    F["compact_eq"] = (comp1 == comp2).astype(np.float32)
    for k in ["first_tok", "meta_first", "sdx_first", "house", "postal", "unit", "street", "legal"]:
        F[f"{k}_eq"] = _eq_nonempty(s1col(k), reccol(k))
    ini1, ini2 = s1col("initials"), reccol("initials")
    F["acronym"] = (((ini1 != "") & (ini1 == comp2)) | ((ini2 != "") & (ini2 == comp1))).astype(np.float32)

    # ---- lengths / missingness / name frequency: per record, then gathered ------------
    cc, fc = idx.core_count, idx.first_count

    def per_row(core, core_toks, addr, first):
        return np.stack([
            np.fromiter((len(x) for x in core), np.float32, len(core)),
            np.fromiter((len(x) for x in core_toks), np.float32, len(core)),
            np.fromiter((len(x) for x in addr), np.float32, len(core)),
            np.log1p(np.fromiter((cc.get(x, 0) for x in core), np.float32, len(core))),
            np.log1p(np.fromiter((fc.get(x, 0) for x in first), np.float32, len(core))),
        ], axis=1)

    R1 = per_row(idx.col("core")[u1], core1, idx.col("addr_n")[u1], idx.col("first_tok")[u1])[ja]
    R2 = per_row(_objarr(rc["core"].tolist())[u2], core2, _objarr(rc["addr_n"].tolist())[u2],
                 _objarr(rc["first_tok"].tolist())[u2])[jb]
    l1, l2 = R1[:, 0], R2[:, 0]
    F["name_len1"], F["name_len2"] = l1, l2
    F["name_len_ratio"] = np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1)
    F["name_ntok1"], F["name_ntok2"] = R1[:, 1], R2[:, 1]
    F["addr_len1"], F["addr_len2"] = R1[:, 2], R2[:, 2]
    F["s1_name_freq"], F["rec_name_freq"] = R1[:, 3], R2[:, 3]   # "chain-ness" among the country's S1
    F["s1_first_freq"] = R1[:, 4]

    # ---- context: this pair vs the record's competing candidates --------------------------
    F["n_cand_rec"] = cand.groupby("rec")["s1"].transform("size").to_numpy().astype(np.float32)
    F["combo"] = (0.4 * F["cos_full_c"] + 0.3 * np.nan_to_num(F["tok_full_wjac"])
                  + 0.3 * np.nan_to_num(F["name_tset"]) / 100).astype(np.float32)
    for sname in ["combo", "cos_name_c", "cos_full_c", "name_tset", "key_score", "cheap", "name_full_ratio"]:
        x = np.nan_to_num(np.asarray(F[sname], dtype=np.float32))
        F[f"{sname}_rank_rec"], F[f"{sname}_gap_rec"] = _group_context(ib, x)

    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in F.items()})
