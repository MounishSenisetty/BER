"""Pairwise feature extraction for candidate pairs.

Everything is vectorised over pairs:
  * TF-IDF cosines         -> sparse row-wise products (scipy)
  * token-set overlaps     -> idf-weighted incidence matrices, row-wise products (scipy)
  * fuzzy string metrics   -> rapidfuzz.process.cpdist (C++, multi-threaded, element-wise)
  * exact / phonetic flags -> numpy comparisons of per-record precomputed keys
  * context features       -> pandas groupby ranks / margins within each record and each S1 group

No per-pair Python loops, so a few million pairs featurise in well under a minute.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .blocking import BLOCKERS, Spaces, rowwise_dot
from .utils import timer


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def _incidence(l1: Sequence[List[str]], l2: Sequence[List[str]]):
    """Binary token-incidence CSR matrices over a shared vocabulary + smoothed idf vector."""
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
    df = np.asarray(B1.sum(0)).ravel() + np.asarray(B2.sum(0)).ravel()
    n = len(l1) + len(l2)
    idf = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)
    return B1, B2, idf


def _overlap_feats(prefix: str, l1, l2, ia, ib, weighted: bool = True) -> Dict[str, np.ndarray]:
    B1, B2, idf = _incidence(l1, l2)
    cnt = rowwise_dot(B1, B2, ia, ib)
    n1 = np.asarray(B1.sum(1)).ravel()[ia]
    n2 = np.asarray(B2.sum(1)).ravel()[ib]
    both_empty = (n1 == 0) & (n2 == 0)
    out = {f"{prefix}_common": cnt,
           f"{prefix}_only1": np.where(both_empty, np.nan, n1 - cnt),
           f"{prefix}_only2": np.where(both_empty, np.nan, n2 - cnt)}
    if weighted:
        W1 = B1.multiply(idf).tocsr()
        W2 = B2.multiply(idf).tocsr()
        inter = rowwise_dot(W1, B2, ia, ib)
        w1 = np.asarray(W1.sum(1)).ravel()[ia]
        w2 = np.asarray(W2.sum(1)).ravel()[ib]
        union = w1 + w2 - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{prefix}_wjac"] = np.where(union > 0, inter / union, np.nan)
            out[f"{prefix}_wcont1"] = np.where(w1 > 0, inter / w1, np.nan)   # share of S1 covered
            out[f"{prefix}_wcont2"] = np.where(w2 > 0, inter / w2, np.nan)   # share of record covered
            out[f"{prefix}_winter"] = inter
            # absolute idf mass left unexplained on each side ("Annex", "Outlet", "Express"...)
            out[f"{prefix}_wonly1"] = w1 - inter
            out[f"{prefix}_wonly2"] = w2 - inter
            out[f"{prefix}_wmaxonly"] = np.maximum(w1, w2) - inter
    return out


def _cp(a: np.ndarray, b: np.ndarray, scorer, n_jobs: int) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=n_jobs, dtype=np.float32)


def _eq_nonempty(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 equal, 0 different, NaN if either side is missing."""
    miss = (a == "") | (b == "")
    return np.where(miss, np.nan, (a == b).astype(np.float32))


def _group_context(g: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rank (1 = best) of x inside its group and margin to the best *other* member."""
    df = pd.DataFrame({"g": g, "x": x})
    grp = df.groupby("g")["x"]
    rank = grp.rank(method="first", ascending=False).to_numpy()
    top1 = grp.transform("max").to_numpy()
    second = df["x"].where(rank == 2).groupby(df["g"]).transform("max").to_numpy()
    best_other = np.where(rank == 1, second, top1)
    return rank.astype(np.float32), (x - best_other).astype(np.float32)


# ----------------------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------------------
def compute_features(s1: pd.DataFrame, rec: pd.DataFrame, cand: pd.DataFrame, spaces: Spaces,
                     n_jobs: int = -1) -> pd.DataFrame:
    ia = cand["s1_idx"].to_numpy()
    ib = cand["rec_idx"].to_numpy()
    F: Dict[str, np.ndarray] = {}

    def col(df, c, idx):
        return np.asarray(df[c].tolist(), dtype=object)[idx]

    with timer(f"Features for {len(cand)} pairs"):
        # ---- blocking provenance ------------------------------------------------------
        for c in BLOCKERS + ["n_blockers", "knn_name_rank", "knn_full_rank", "cheap", "cheap_rank",
                             "cos_name_c", "cos_full_c"]:
            F[c] = cand[c].to_numpy()
        F["rec_source"] = rec["source"].to_numpy()[ib].astype(np.int8)

        # ---- tf-idf cosines -----------------------------------------------------------
        F["cos_addr_c"] = rowwise_dot(spaces.addr_c1, spaces.addr_c2, ia, ib)
        F["cos_name_w"] = rowwise_dot(spaces.name_w1, spaces.name_w2, ia, ib)
        F["cos_addr_w"] = rowwise_dot(spaces.addr_w1, spaces.addr_w2, ia, ib)

        # ---- token-set overlaps -------------------------------------------------------
        F.update(_overlap_feats("tok_name", s1["core_toks"].tolist(), rec["core_toks"].tolist(), ia, ib))
        F.update(_overlap_feats("tok_addr", s1["addr_toks"].tolist(), rec["addr_toks"].tolist(), ia, ib))
        full1 = [a + b for a, b in zip(s1["core_toks"], s1["addr_toks"])]
        full2 = [a + b for a, b in zip(rec["core_toks"], rec["addr_toks"])]
        F.update(_overlap_feats("tok_full", full1, full2, ia, ib))
        F.update(_overlap_feats("num_addr", s1["addr_nums"].tolist(), rec["addr_nums"].tolist(), ia, ib,
                                weighted=False))
        F.update(_overlap_feats("num_name", s1["name_nums"].tolist(), rec["name_nums"].tolist(), ia, ib,
                                weighted=False))

        # ---- fuzzy string metrics -----------------------------------------------------
        c1, c2 = col(s1, "core", ia), col(rec, "core", ib)
        a1, a2 = col(s1, "addr_n", ia), col(rec, "addr_n", ib)
        n_empty = (c1 == "") | (c2 == "")
        a_empty = (a1 == "") | (a2 == "")
        for nm, sc in [("name_ratio", fuzz.ratio), ("name_pratio", fuzz.partial_ratio),
                       ("name_tsort", fuzz.token_sort_ratio), ("name_tset", fuzz.token_set_ratio),
                       ("name_jw", JaroWinkler.normalized_similarity)]:
            F[nm] = np.where(n_empty, np.nan, _cp(c1, c2, sc, n_jobs))
        F["name_full_ratio"] = _cp(col(s1, "name_n", ia), col(rec, "name_n", ib), fuzz.ratio, n_jobs)
        for nm, sc in [("addr_ratio", fuzz.ratio), ("addr_pratio", fuzz.partial_ratio),
                       ("addr_tset", fuzz.token_set_ratio), ("addr_tsort", fuzz.token_sort_ratio)]:
            F[nm] = np.where(a_empty, np.nan, _cp(a1, a2, sc, n_jobs))
        F["full_tset"] = _cp(col(s1, "full", ia), col(rec, "full", ib), fuzz.token_set_ratio, n_jobs)
        F["phon_ratio"] = _cp(col(s1, "phon", ia), col(rec, "phon", ib), fuzz.ratio, n_jobs)
        F["compact_jw"] = _cp(col(s1, "compact", ia), col(rec, "compact", ib),
                              JaroWinkler.normalized_similarity, n_jobs)
        F["first_jw"] = _cp(col(s1, "first_tok", ia), col(rec, "first_tok", ib),
                            JaroWinkler.normalized_similarity, n_jobs)
        # record name appearing inside the S1 address or vice versa (field-swap noise)
        F["name2_in_addr1"] = np.where(n_empty | (a1 == ""), np.nan, _cp(c2, a1, fuzz.partial_ratio, n_jobs))

        # ---- exact / phonetic / parsed-address flags ----------------------------------
        comp1, comp2 = col(s1, "compact", ia), col(rec, "compact", ib)
        F["core_eq"] = (c1 == c2).astype(np.float32)
        F["compact_eq"] = (comp1 == comp2).astype(np.float32)
        for k in ["first_tok", "meta_first", "sdx_first", "house", "postal", "unit", "street"]:
            F[f"{k}_eq"] = _eq_nonempty(col(s1, k, ia), col(rec, k, ib))
        ini1, ini2 = col(s1, "initials", ia), col(rec, "initials", ib)
        F["acronym"] = (((ini1 != "") & (ini1 == comp2)) | ((ini2 != "") & (ini2 == comp1))).astype(np.float32)

        # ---- lengths / missingness ----------------------------------------------------
        l1 = np.fromiter((len(x) for x in c1), np.float32, len(c1))
        l2 = np.fromiter((len(x) for x in c2), np.float32, len(c2))
        F["name_len1"], F["name_len2"] = l1, l2
        F["name_len_ratio"] = np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1)
        F["name_ntok1"] = s1["core_toks"].map(len).to_numpy()[ia].astype(np.float32)
        F["name_ntok2"] = rec["core_toks"].map(len).to_numpy()[ib].astype(np.float32)
        F["addr_len1"] = s1["addr_n"].str.len().to_numpy()[ia].astype(np.float32)
        F["addr_len2"] = rec["addr_n"].str.len().to_numpy()[ib].astype(np.float32)

        # ---- name frequency ("chain-ness"): common names must be decided by address ----
        s1_core_cnt = s1["core"].map(s1["core"].value_counts()).to_numpy()
        rec_core_cnt = rec["core"].map(rec["core"].value_counts()).to_numpy()
        s1_first_cnt = s1["first_tok"].map(s1["first_tok"].value_counts()).to_numpy()
        F["s1_name_freq"] = np.log1p(s1_core_cnt[ia]).astype(np.float32)
        F["rec_name_freq"] = np.log1p(rec_core_cnt[ib]).astype(np.float32)
        F["s1_first_freq"] = np.log1p(s1_first_cnt[ia]).astype(np.float32)

        # ---- context: how does this pair compare with competing candidates? -----------
        F["n_cand_rec"] = cand.groupby("rec_idx")["s1_idx"].transform("size").to_numpy().astype(np.float32)
        F["n_cand_s1"] = cand.groupby("s1_idx")["rec_idx"].transform("size").to_numpy().astype(np.float32)
        F["combo"] = (0.4 * F["cos_full_c"] + 0.3 * np.nan_to_num(F["tok_full_wjac"])
                      + 0.3 * np.nan_to_num(F["name_tset"]) / 100).astype(np.float32)
        for sname in ["combo", "cos_name_c", "cos_full_c", "name_tset"]:
            x = np.nan_to_num(np.asarray(F[sname], dtype=np.float32))
            F[f"{sname}_rank_rec"], F[f"{sname}_gap_rec"] = _group_context(ib, x)
            F[f"{sname}_rank_s1"], F[f"{sname}_gap_s1"] = _group_context(ia, x)

    feats = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in F.items()})
    return feats
