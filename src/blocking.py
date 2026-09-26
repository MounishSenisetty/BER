"""Scalable blocking: per-country Source 1 indexes queried by chunks of Source 2/3 records.

Direction: every S2/S3 record *queries* the Source 1 index of its country. A noisy record belongs
to at most one S1 entity, while an S1 entity may own many records.

For each country partition the Source 1 side builds
  * hashed char 3-gram TF-IDF spaces (core name / name+address / address), idf fitted on S1
  * dense vectors = TruncatedSVD(TF-IDF) -> two FAISS inner-product indexes (name, name+address)
  * a sparse key index over rare tokens, token pairs and phonetic / address compound keys
    (keys shared by more than `key_max_df` S1 rows are dropped as unselective)

Each record chunk is searched with the three generators (dense-name kNN, dense-full kNN, key
overlap); the union is scored with a cheap similarity, and the top `max_candidates_per_record`
per record are kept. That capped set is exactly what the classifier scores and what
candidate_pairs.tsv contains.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer

from .config import BlockingConfig
from .utils import LOG

try:
    import faiss
except ImportError:  # small data still works with brute-force numpy search
    faiss = None

BLOCKERS = ["b_knn_name", "b_knn_full", "b_key"]


# ----------------------------------------------------------------------------------------------
# sparse / dense helpers
# ----------------------------------------------------------------------------------------------
def rowwise_dot(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray,
                chunk: int = 1_000_000) -> np.ndarray:
    """dot(A[ia[k]], B[ib[k]]) for every k (cosine when rows are L2-normalised)."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        e = min(s + chunk, len(ia))
        out[s:e] = np.asarray(A[ia[s:e]].multiply(B[ib[s:e]]).sum(axis=1)).ravel()
    return out


def dense_rowdot(A: np.ndarray, B: np.ndarray, ia: np.ndarray, ib: np.ndarray,
                 chunk: int = 65_536) -> np.ndarray:
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        e = min(s + chunk, len(ia))
        out[s:e] = np.einsum("ij,ij->i", A[ia[s:e]], B[ib[s:e]])
    return out


def csr_topk(S: sp.csr_matrix, k: int):
    """Vectorised per-row top-k of a CSR matrix -> (row, col, value, rank)."""
    S = S.tocsr()
    rows = np.repeat(np.arange(S.shape[0], dtype=np.int64), np.diff(S.indptr))
    order = np.lexsort((-S.data, rows))
    rows_s = rows[order]
    rank = np.arange(len(order), dtype=np.int64) - S.indptr[rows_s]
    keep = rank < k
    return rows_s[keep], S.indices[order][keep].astype(np.int64), S.data[order][keep], (rank[keep] + 1)


def _l2_normalize_rows(X: sp.csr_matrix) -> sp.csr_matrix:
    sq = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    inv = np.where(sq > 0, 1.0 / np.sqrt(np.maximum(sq, 1e-12)), 0.0).astype(np.float32)
    return sp.diags(inv).dot(X).tocsr()


def _n_jobs(n: int) -> int:
    import os
    return (os.cpu_count() or 1) if n is None or n < 1 else n


# ----------------------------------------------------------------------------------------------
# TF-IDF + SVD space
# ----------------------------------------------------------------------------------------------
class CharSpace:
    """Hashed char n-gram TF-IDF (idf fitted on S1) with an optional dense SVD projection."""

    def __init__(self, cfg: BlockingConfig, dense: bool, n_jobs: int = -1):
        self.cfg, self.dense, self.n_jobs = cfg, dense, _n_jobs(n_jobs)
        self.hv = HashingVectorizer(analyzer="char_wb", ngram_range=tuple(cfg.ngram_range),
                                    n_features=cfg.hash_features, alternate_sign=False, norm=None,
                                    dtype=np.float32)

    def _counts(self, texts: List[str]) -> sp.csr_matrix:
        if len(texts) < 100_000 or self.n_jobs == 1:
            return self.hv.transform(texts).tocsr()
        step = int(math.ceil(len(texts) / self.n_jobs))
        parts = Parallel(n_jobs=self.n_jobs)(delayed(self.hv.transform)(texts[i:i + step])
                                             for i in range(0, len(texts), step))
        return sp.vstack(parts).tocsr()

    def _weight(self, X: sp.csr_matrix) -> sp.csr_matrix:
        X = X.copy()
        X.data = (1.0 + np.log(X.data)) * self.idf[X.indices]
        return _l2_normalize_rows(X)

    def _project(self, Xw: sp.csr_matrix) -> np.ndarray:
        m = self.colmap[Xw.indices]
        keep = m >= 0
        rows = np.repeat(np.arange(Xw.shape[0]), np.diff(Xw.indptr))
        indptr = np.r_[0, np.cumsum(np.bincount(rows[keep], minlength=Xw.shape[0]))]
        Xa = sp.csr_matrix((Xw.data[keep], m[keep], indptr), shape=(Xw.shape[0], len(self.active)))
        D = np.asarray(Xa @ self.comp_T, dtype=np.float32)
        nrm = np.linalg.norm(D, axis=1, keepdims=True)
        return D / np.maximum(nrm, 1e-8)

    def fit_transform(self, texts: List[str], seed: int = 0):
        X = self._counts(texts)
        N = X.shape[0]
        df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float32)
        self.idf = (np.log((N + 1) / (df + 1)) + 1).astype(np.float32)
        Xw = self._weight(X)
        D = None
        if self.dense:
            self.active = np.flatnonzero(df > 0)
            self.colmap = np.full(X.shape[1], -1, dtype=np.int64)
            self.colmap[self.active] = np.arange(len(self.active))
            rng = np.random.default_rng(seed)
            rows = rng.choice(N, size=min(N, self.cfg.svd_fit_rows), replace=False)
            Xs = Xw[rows][:, self.active]
            dim = int(max(2, min(self.cfg.svd_dim, Xs.shape[1] - 1, Xs.shape[0] - 1)))
            svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=4, random_state=seed)
            svd.fit(Xs)
            self.comp_T = svd.components_.T.astype(np.float32)
            D = self._project(Xw)
        return Xw, D

    def transform(self, texts: List[str]):
        Xw = self._weight(self._counts(texts))
        return Xw, (self._project(Xw) if self.dense else None)


# ----------------------------------------------------------------------------------------------
# key index
# ----------------------------------------------------------------------------------------------
def make_keys(df: pd.DataFrame) -> List[List[str]]:
    """Blocking keys per record. Name keys catch lexical overlap; address bigram and
    acronym x address-number keys catch records whose name is a trade name or an acronym."""
    out = []
    for ct, comp, ho, st, po, ft, mf, at, ini, nums in zip(
            df["core_toks"], df["compact"], df["house"], df["street"], df["postal"], df["first_tok"],
            df["meta_first"], df["addr_toks"], df["initials"], df["addr_nums"]):
        toks = list(dict.fromkeys(t for t in ct if len(t) >= 2 and not t.isdigit()))
        ks = ["t|" + t for t in toks]
        head = toks[:4]
        ks += [f"p|{a}|{b}" if a < b else f"p|{b}|{a}" for i, a in enumerate(head) for b in head[i + 1:]]
        if len(comp) >= 4:
            ks.append("c|" + comp)
        if ho and st:
            ks.append(f"hs|{ho}|{st}")
        if po and ft:
            ks.append(f"pf|{po}|{ft}")
        if mf and po:
            ks.append(f"mp|{mf}|{po}")
        if mf and ho:
            ks.append(f"mh|{mf}|{ho}")
        if st and ft:
            ks.append(f"sf|{st}|{ft}")
        ks += [f"ab|{a}|{b}" for a, b in zip(at[:13], at[1:13]) if len(a) + len(b) >= 5]
        acr = {x for x in (ini, comp if 2 <= len(comp) <= 5 and comp.isalpha() else "") if x}
        ks += [f"ac|{x}|{n}" for x in acr for n in nums[:3]]
        out.append(ks)
    return out


# ----------------------------------------------------------------------------------------------
# per-country Source 1 index
# ----------------------------------------------------------------------------------------------
@dataclass
class RecordMats:
    name_X: sp.csr_matrix
    full_X: sp.csr_matrix
    addr_X: sp.csr_matrix
    name_D: np.ndarray
    full_D: np.ndarray


def _build_faiss(D: np.ndarray, cfg: BlockingConfig, seed: int):
    N, d = D.shape
    if faiss is None:
        return None
    if N <= cfg.exact_knn_below:
        index = faiss.IndexFlatIP(d)
        index.add(D)
        return index
    nlist = int(min(65536, max(16, min(4 * math.sqrt(N), N / 50))))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(seed)
    index.train(D[rng.choice(N, size=min(N, nlist * 40), replace=False)])
    index.add(D)
    index.nprobe = cfg.faiss_nprobe
    return _maybe_gpu(index, cfg)


def _maybe_gpu(index, cfg: BlockingConfig):
    """Move an IVF index to the GPU(s) when a GPU build of FAISS sees one; otherwise keep the CPU
    index. Results are the same search (same nprobe), only faster."""
    if not getattr(cfg, "use_gpu", True) or not hasattr(faiss, "get_num_gpus"):
        return index
    try:
        if faiss.get_num_gpus() < 1:
            return index
        gpu = faiss.index_cpu_to_all_gpus(index)
        faiss.GpuParameterSpace().set_index_parameter(gpu, "nprobe", cfg.faiss_nprobe)
        return gpu
    except Exception as e:  # any GPU problem -> silently stay on CPU
        LOG.warning("FAISS GPU unavailable (%s); using CPU index", e)
        return index


def _search(index, D_base: np.ndarray, Q: np.ndarray, k: int):
    k = min(k, D_base.shape[0])
    if index is not None:
        sims, ids = index.search(np.ascontiguousarray(Q, dtype=np.float32), k)
    else:  # brute force fallback (small partitions only)
        sims = np.empty((len(Q), k), np.float32)
        ids = np.empty((len(Q), k), np.int64)
        for s in range(0, len(Q), 4096):
            S = Q[s:s + 4096] @ D_base.T
            top = np.argpartition(-S, k - 1, axis=1)[:, :k]
            ts = np.take_along_axis(S, top, axis=1)
            o = np.argsort(-ts, axis=1)
            ids[s:s + 4096] = np.take_along_axis(top, o, axis=1)
            sims[s:s + 4096] = np.take_along_axis(ts, o, axis=1)
    rows = np.repeat(np.arange(len(Q), dtype=np.int64), ids.shape[1])
    rank = np.tile(np.arange(1, ids.shape[1] + 1, dtype=np.int64), len(Q))
    ids = ids.ravel()
    ok = ids >= 0
    return rows[ok], ids[ok].astype(np.int64), rank[ok]


def _rank_within(g: np.ndarray, x: np.ndarray) -> np.ndarray:
    """1-based descending rank of x within groups g."""
    order = np.lexsort((-x, g))
    gs = g[order]
    start = np.r_[True, gs[1:] != gs[:-1]]
    first = np.flatnonzero(start)
    pos = np.arange(len(g)) - first[np.cumsum(start) - 1]
    rank = np.empty(len(g), dtype=np.int64)
    rank[order] = pos + 1
    return rank


class CountryIndex:
    """Everything about one country's Source 1 partition needed for blocking and features."""

    def __init__(self, s1c: pd.DataFrame, cfg: BlockingConfig, n_jobs: int = -1, seed: int = 0):
        if faiss is not None:
            faiss.omp_set_num_threads(_n_jobs(n_jobs))
        self.cfg, self.n_jobs = cfg, _n_jobs(n_jobs)
        self.s1 = s1c.reset_index(drop=True)
        self.N = len(self.s1)
        self.name_space = CharSpace(cfg, dense=True, n_jobs=n_jobs)
        self.full_space = CharSpace(cfg, dense=True, n_jobs=n_jobs)
        self.addr_space = CharSpace(cfg, dense=False, n_jobs=n_jobs)
        self.name_X, self.name_D = self.name_space.fit_transform(self.s1["core"].tolist(), seed)
        self.full_X, self.full_D = self.full_space.fit_transform(self.s1["full"].tolist(), seed)
        self.addr_X, _ = self.addr_space.fit_transform(self.s1["addr_n"].tolist(), seed)
        self.name_index = _build_faiss(self.name_D, cfg, seed)
        self.full_index = _build_faiss(self.full_D, cfg, seed)

        self.hasher = FeatureHasher(n_features=cfg.key_hash_features, input_type="string", alternate_sign=False)
        K = self.hasher.transform(make_keys(self.s1)).tocsr()
        K.data[:] = 1.0
        df = np.bincount(K.indices, minlength=K.shape[1])
        self.key_ok = (df >= 1) & (df <= cfg.key_max_df)
        idf = np.zeros(K.shape[1], dtype=np.float32)
        idf[self.key_ok] = np.log((self.N + 1) / df[self.key_ok]).astype(np.float32)
        K.data = idf[K.indices]
        K.eliminate_zeros()
        self.key_W = K
        self.key_WT = K.T.tocsr()

        # token idf tables (features) and name frequency ("chain-ness")
        self.tok_idf = self._idf(self.s1["core_toks"])
        self.addr_idf = self._idf(self.s1["addr_toks"])
        self.core_count = self.s1["core"].value_counts().to_dict()
        self.first_count = self.s1["first_tok"].value_counts().to_dict()
        # object arrays for fast gathers in feature extraction
        self.cols: Dict[str, np.ndarray] = {}

    def _idf(self, lists) -> Dict[str, float]:
        from collections import Counter
        c = Counter(t for toks in lists for t in set(toks))
        n = len(lists)
        return {t: math.log((n + 1) / (v + 1)) + 1 for t, v in c.items()}

    def col(self, name: str) -> np.ndarray:
        if name not in self.cols:
            self.cols[name] = np.asarray(self.s1[name].tolist() + [None], dtype=object)[:-1]
        return self.cols[name]

    # ------------------------------------------------------------------------------------------
    def query(self, rc: pd.DataFrame) -> Tuple[pd.DataFrame, RecordMats]:
        cfg = self.cfg
        name_X, name_D = self.name_space.transform(rc["core"].tolist())
        full_X, full_D = self.full_space.transform(rc["full"].tolist())
        addr_X, _ = self.addr_space.transform(rc["addr_n"].tolist())
        mats = RecordMats(name_X, full_X, addr_X, name_D, full_D)
        n = len(rc)

        r1, s1, k1 = _search(self.name_index, self.name_D, name_D, cfg.name_topk)
        # records without an address can only be found by name: search much wider for them
        no_addr = (rc["addr_n"].to_numpy(dtype=object) == "")
        wide_k = getattr(cfg, "empty_addr_topk", 0)
        if wide_k > cfg.name_topk and no_addr.any():
            rows = np.flatnonzero(no_addr)
            rw, sw, kw = _search(self.name_index, self.name_D, name_D[rows], wide_k)
            r1, s1, k1 = np.concatenate([r1, rows[rw]]), np.concatenate([s1, sw]), np.concatenate([k1, kw])
        r2, s2, k2 = _search(self.full_index, self.full_D, full_D, cfg.full_topk)

        Q = self.hasher.transform(make_keys(rc)).tocsr()
        Q.data[:] = 1.0
        Q.data *= self.key_ok[Q.indices]
        Q.eliminate_zeros()
        r3, s3, k3 = [], [], []
        for st in range(0, n, 20_000):
            S = (Q[st:st + 20_000] @ self.key_WT).tocsr()
            rr, cc, _, kk = csr_topk(S, cfg.key_topk)
            r3.append(rr + st), s3.append(cc), k3.append(kk)
        r3 = np.concatenate(r3) if r3 else np.zeros(0, np.int64)
        s3 = np.concatenate(s3) if s3 else np.zeros(0, np.int64)
        k3 = np.concatenate(k3) if k3 else np.zeros(0, np.int64)

        # ---- union (vectorised de-duplication on a combined int64 key) ----
        rec = np.concatenate([r1, r2, r3])
        s1i = np.concatenate([s1, s2, s3])
        src = np.concatenate([np.zeros(len(r1), np.int8), np.ones(len(r2), np.int8), np.full(len(r3), 2, np.int8)])
        rnk = np.concatenate([k1, k2, k3]).astype(np.int32)
        key = rec * self.N + s1i
        uniq, inv = np.unique(key, return_inverse=True)
        m = len(uniq)
        big = np.int32(1_000)
        ranks = np.full((3, m), big, dtype=np.int32)
        for b in range(3):
            sel = src == b
            np.minimum.at(ranks[b], inv[sel], rnk[sel])
        cand = pd.DataFrame({"rec": (uniq // self.N).astype(np.int64), "s1": (uniq % self.N).astype(np.int64)})
        cand["b_knn_name"] = (ranks[0] < big).astype(np.int8)
        cand["b_knn_full"] = (ranks[1] < big).astype(np.int8)
        cand["b_key"] = (ranks[2] < big).astype(np.int8)
        cand["knn_name_rank"] = np.minimum(ranks[0], max(cfg.name_topk, wide_k) + 1)
        cand["knn_full_rank"] = np.minimum(ranks[1], cfg.full_topk + 1)
        cand["key_rank"] = np.minimum(ranks[2], cfg.key_topk + 1)
        cand["n_blockers"] = (cand["b_knn_name"] + cand["b_knn_full"] + cand["b_key"]).astype(np.int8)

        ia, ib = cand["s1"].to_numpy(), cand["rec"].to_numpy()
        cand["cos_name_d"] = dense_rowdot(self.name_D, name_D, ia, ib)
        cand["cos_full_d"] = dense_rowdot(self.full_D, full_D, ia, ib)
        cand["key_score"] = rowwise_dot(self.key_W, Q, ia, ib)
        cand["cos_addr_c"] = rowwise_dot(self.addr_X, addr_X, ia, ib)
        cand["cheap"] = (0.3 * cand["cos_name_d"] + 0.3 * cand["cos_full_d"] + 0.25 * cand["cos_addr_c"]
                         + 0.15 * np.tanh(cand["key_score"] / 10.0) + 0.01 * cand["n_blockers"]).astype(np.float32)

        # keep the top-K by the cheap score, plus a few address / key specialists per record so that
        # trade names and acronyms (name says nothing, address says everything) are not capped away
        rec_arr = cand["rec"].to_numpy()
        cand["cheap_rank"] = _rank_within(rec_arr, cand["cheap"].to_numpy()).astype(np.int16)
        addr_rank = _rank_within(rec_arr, cand["cos_addr_c"].to_numpy())
        cap = np.where(no_addr[rec_arr], getattr(cfg, "empty_addr_keep", cfg.max_candidates_per_record),
                       cfg.max_candidates_per_record)
        keep = ((cand["cheap_rank"].to_numpy() <= cap)
                | ((addr_rank <= cfg.addr_keep) & (cand["cos_addr_c"].to_numpy() >= cfg.addr_keep_min_cos))
                | (cand["key_rank"].to_numpy() <= cfg.key_keep))
        self.last_union = (rec_arr, cand["s1"].to_numpy())      # pre-cap union (training diagnostics)
        cand = cand[keep].sort_values(["rec", "cheap"], ascending=[True, False], kind="stable")
        return cand.reset_index(drop=True), mats
