"""High-recall blocking: union of complementary candidate generators, then a cheap re-rank + cap.

Direction: every Source 2 / Source 3 record *queries* the Source 1 reference set. This is the
natural direction because a noisy record belongs to at most one S1 entity, while an S1 entity
may own many records.

Generators (each emits (rec_idx, s1_idx) pairs + a flag column used later as a feature):
  1. TF-IDF char n-gram kNN on the core name           (typos, abbreviations, word order)
  2. TF-IDF char n-gram kNN on core name + address      (disambiguates chains / generic names)
  3. Rare-name-token inverted index                     (exact token overlap, IDF-capped buckets)
  4. Phonetic compound keys: metaphone(first token) x {postal, house no.}, soundex x postal
  5. Address compound keys: house no. x street token, postal x first name token
  6. MinHash LSH over name+address token sets           (datasketch)

Run standalone to export candidate_pairs.tsv:
    python -m src.blocking --data-dir data/test --out output/candidate_pairs.tsv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import BlockingConfig
from .utils import LOG, timer

BLOCKERS = ["b_knn_name", "b_knn_full", "b_token", "b_phon", "b_addr", "b_lsh"]


# ----------------------------------------------------------------------------------------
# Vector spaces (shared with feature extraction)
# ----------------------------------------------------------------------------------------
@dataclass
class Spaces:
    name_c1: sp.csr_matrix   # char n-gram tfidf of core names, S1
    name_c2: sp.csr_matrix   # ... records
    full_c1: sp.csr_matrix
    full_c2: sp.csr_matrix
    addr_c1: sp.csr_matrix
    addr_c2: sp.csr_matrix
    name_w1: sp.csr_matrix   # word tfidf of core names
    name_w2: sp.csr_matrix
    addr_w1: sp.csr_matrix
    addr_w2: sp.csr_matrix


def _fit_pair(a: List[str], b: List[str], **kw):
    vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kw)
    try:
        vec.fit(a + b)
    except ValueError:  # empty vocabulary (e.g. no addresses at all)
        z1 = sp.csr_matrix((len(a), 1), dtype=np.float32)
        z2 = sp.csr_matrix((len(b), 1), dtype=np.float32)
        return z1, z2
    return vec.transform(a).tocsr(), vec.transform(b).tocsr()


def build_spaces(s1: pd.DataFrame, rec: pd.DataFrame, cfg: BlockingConfig) -> Spaces:
    """Fit TF-IDF on S1 + records of the *current* split (unsupervised, no labels)."""
    with timer("Fitting TF-IDF spaces"):
        char_kw = dict(analyzer="char_wb", ngram_range=tuple(cfg.ngram_range), min_df=2,
                       max_df=cfg.tfidf_max_df)
        word_kw = dict(analyzer="word", token_pattern=r"(?u)\b\w+\b", min_df=1)
        n1, n2 = _fit_pair(s1["core"].tolist(), rec["core"].tolist(), **char_kw)
        f1, f2 = _fit_pair(s1["full"].tolist(), rec["full"].tolist(), **char_kw)
        a1, a2 = _fit_pair(s1["addr_n"].tolist(), rec["addr_n"].tolist(), **char_kw)
        w1, w2 = _fit_pair(s1["core"].tolist(), rec["core"].tolist(), **word_kw)
        aw1, aw2 = _fit_pair(s1["addr_n"].tolist(), rec["addr_n"].tolist(), **word_kw)
    return Spaces(n1, n2, f1, f2, a1, a2, w1, w2, aw1, aw2)


def rowwise_dot(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray,
                chunk: int = 500_000) -> np.ndarray:
    """dot(A[ia[k]], B[ib[k]]) for every k -- cosine if rows are L2-normalised."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        e = min(s + chunk, len(ia))
        out[s:e] = np.asarray(A[ia[s:e]].multiply(B[ib[s:e]]).sum(axis=1)).ravel()
    return out


# ----------------------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------------------
def sparse_topk(Q: sp.csr_matrix, R: sp.csr_matrix, k: int, min_sim: float, chunk: int) -> pd.DataFrame:
    """Top-k rows of R for each row of Q by cosine similarity, via chunked sparse matmul."""
    RT = R.T.tocsr()
    qi, ri, rank = [], [], []
    for start in range(0, Q.shape[0], chunk):
        S = (Q[start:start + chunk] @ RT).tocsr()
        for i in range(S.shape[0]):
            lo, hi = S.indptr[i], S.indptr[i + 1]
            if hi == lo:
                continue
            d = S.data[lo:hi]
            c = S.indices[lo:hi]
            keep = d >= min_sim
            d, c = d[keep], c[keep]
            if len(d) > k:
                sel = np.argpartition(-d, k - 1)[:k]
                d, c = d[sel], c[sel]
            order = np.argsort(-d, kind="stable")
            qi.append(np.full(len(order), start + i, dtype=np.int64))
            ri.append(c[order].astype(np.int64))
            rank.append(np.arange(1, len(order) + 1, dtype=np.int32))
    if not qi:
        return pd.DataFrame({"rec_idx": [], "s1_idx": [], "rank": []}, dtype=np.int64)
    return pd.DataFrame({"rec_idx": np.concatenate(qi), "s1_idx": np.concatenate(ri),
                         "rank": np.concatenate(rank)})


def pairs_from_keys(k1: pd.DataFrame, k2: pd.DataFrame, max_bucket: int) -> pd.DataFrame:
    """Equi-join on a blocking key. k1: (s1_idx, key), k2: (rec_idx, key).
    Keys whose S1 bucket exceeds `max_bucket` are dropped (too unselective)."""
    k1 = k1[k1["key"] != ""].drop_duplicates()
    k2 = k2[k2["key"] != ""].drop_duplicates()
    size = k1.groupby("key").size()
    k1 = k1[k1["key"].isin(size.index[size <= max_bucket])]
    m = k2.merge(k1, on="key", how="inner")
    return m[["rec_idx", "s1_idx"]].drop_duplicates()


def _explode(idx_name: str, values: List[List[str]]) -> pd.DataFrame:
    lens = np.fromiter((len(v) for v in values), dtype=np.int64, count=len(values))
    return pd.DataFrame({idx_name: np.repeat(np.arange(len(values)), lens),
                         "key": [t for v in values for t in v]})


def _compound_keys(df: pd.DataFrame, kind: str) -> List[List[str]]:
    out = []
    if kind == "phon":
        for mf, sx, po, ho in zip(df["meta_first"], df["sdx_first"], df["postal"], df["house"]):
            ks = []
            if mf and po:
                ks.append(f"mp|{mf}|{po}")
            if mf and ho:
                ks.append(f"mh|{mf}|{ho}")
            if sx and po:
                ks.append(f"sp|{sx}|{po}")
            out.append(ks)
    elif kind == "addr":
        for ho, st, po, ft in zip(df["house"], df["street"], df["postal"], df["first_tok"]):
            ks = []
            if ho and st:
                ks.append(f"hs|{ho}|{st}")
            if po and ft:
                ks.append(f"pf|{po}|{ft}")
            out.append(ks)
    return out


def _name_tokens(df: pd.DataFrame) -> List[List[str]]:
    return [sorted({t for t in toks if len(t) >= 2}) for toks in df["core_toks"]]


def lsh_pairs(s1: pd.DataFrame, rec: pd.DataFrame, cfg: BlockingConfig) -> pd.DataFrame:
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        LOG.warning("datasketch not installed -> skipping LSH blocker")
        return pd.DataFrame({"rec_idx": [], "s1_idx": []}, dtype=np.int64)

    def sets(df):
        return [[t.encode("utf8") for t in set(a) | set(b)] or [b"__empty__"]
                for a, b in zip(df["core_toks"], df["addr_toks"])]

    lsh = MinHashLSH(threshold=cfg.lsh_threshold, num_perm=cfg.lsh_num_perm)
    mh1 = MinHash.bulk(sets(s1), num_perm=cfg.lsh_num_perm)
    with lsh.insertion_session() as session:
        for i, m in enumerate(mh1):
            session.insert(i, m, check_duplication=False)
    mh2 = MinHash.bulk(sets(rec), num_perm=cfg.lsh_num_perm)
    ri, si = [], []
    for j, m in enumerate(mh2):
        for i in lsh.query(m):
            ri.append(j)
            si.append(i)
    return pd.DataFrame({"rec_idx": np.asarray(ri, dtype=np.int64), "s1_idx": np.asarray(si, dtype=np.int64)})


# ----------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------
def generate_candidates(s1: pd.DataFrame, rec: pd.DataFrame, spaces: Spaces,
                        cfg: BlockingConfig) -> pd.DataFrame:
    """Returns one row per candidate (rec_idx, s1_idx) with blocker flags, kNN ranks and a
    cheap similarity score. Output is capped at cfg.max_candidates_per_record per record."""
    parts: Dict[str, pd.DataFrame] = {}
    with timer("Blocking: TF-IDF kNN (name)"):
        knn_n = sparse_topk(spaces.name_c2, spaces.name_c1, cfg.name_topk, cfg.min_sim, cfg.chunk_size)
    with timer("Blocking: TF-IDF kNN (name+address)"):
        knn_f = sparse_topk(spaces.full_c2, spaces.full_c1, cfg.full_topk, cfg.min_sim, cfg.chunk_size)
    with timer("Blocking: inverted indexes"):
        parts["b_token"] = pairs_from_keys(_explode("s1_idx", _name_tokens(s1)),
                                           _explode("rec_idx", _name_tokens(rec)),
                                           cfg.token_max_bucket)
        parts["b_phon"] = pairs_from_keys(_explode("s1_idx", _compound_keys(s1, "phon")),
                                          _explode("rec_idx", _compound_keys(rec, "phon")),
                                          cfg.key_max_bucket)
        parts["b_addr"] = pairs_from_keys(_explode("s1_idx", _compound_keys(s1, "addr")),
                                          _explode("rec_idx", _compound_keys(rec, "addr")),
                                          cfg.key_max_bucket)
    if cfg.use_lsh:
        with timer("Blocking: MinHash LSH"):
            parts["b_lsh"] = lsh_pairs(s1, rec, cfg)

    frames = [knn_n.rename(columns={"rank": "knn_name_rank"}).assign(b_knn_name=1),
              knn_f.rename(columns={"rank": "knn_full_rank"}).assign(b_knn_full=1)]
    for name, df in parts.items():
        LOG.info("  %-7s -> %d raw pairs", name, len(df))
        frames.append(df.assign(**{name: 1}))
    LOG.info("  knn_name-> %d raw pairs | knn_full -> %d raw pairs", len(knn_n), len(knn_f))

    allp = pd.concat(frames, ignore_index=True)
    agg = {c: "max" for c in BLOCKERS if c in allp.columns}
    agg.update({"knn_name_rank": "min", "knn_full_rank": "min"})
    cand = allp.groupby(["rec_idx", "s1_idx"], sort=False).agg(agg).reset_index()
    for c in BLOCKERS:
        cand[c] = cand[c].fillna(0).astype(np.int8) if c in cand.columns else np.int8(0)
    cand["knn_name_rank"] = cand["knn_name_rank"].fillna(cfg.name_topk + 1).astype(np.int32)
    cand["knn_full_rank"] = cand["knn_full_rank"].fillna(cfg.full_topk + 1).astype(np.int32)
    cand["n_blockers"] = cand[BLOCKERS].sum(axis=1).astype(np.int8)

    ia, ib = cand["s1_idx"].to_numpy(), cand["rec_idx"].to_numpy()
    cand["cos_name_c"] = rowwise_dot(spaces.name_c1, spaces.name_c2, ia, ib)
    cand["cos_full_c"] = rowwise_dot(spaces.full_c1, spaces.full_c2, ia, ib)
    cand["cheap"] = (0.55 * cand["cos_name_c"] + 0.45 * cand["cos_full_c"]
                     + 0.02 * cand["n_blockers"]).astype(np.float32)

    before = len(cand)
    cand = cand.sort_values(["rec_idx", "cheap"], ascending=[True, False], kind="stable")
    cand["cheap_rank"] = cand.groupby("rec_idx").cumcount().astype(np.int32) + 1
    cand = cand[cand["cheap_rank"] <= cfg.max_candidates_per_record].reset_index(drop=True)
    LOG.info("Candidates: %d unioned -> %d after cap (%.1f / record, %d records, %d S1 entities)",
             before, len(cand), len(cand) / max(len(rec), 1), len(rec), len(s1))
    return cand


def attach_ids(cand: pd.DataFrame, s1: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    cand = cand.copy()
    cand["source1_id"] = s1["id"].to_numpy()[cand["s1_idx"].to_numpy()]
    cand["candidate_id"] = rec["id"].to_numpy()[cand["rec_idx"].to_numpy()]
    cand["candidate_source"] = rec["source"].to_numpy()[cand["rec_idx"].to_numpy()]
    return cand


def blocking_report(cand: pd.DataFrame, truth: Dict[str, set], rec_ids: Optional[set] = None) -> Dict[str, float]:
    """Pair recall of the candidate set and the macro-F0.5 ceiling it implies."""
    from .utils import macro_f05
    pos = set(zip(cand["source1_id"], cand["candidate_id"]))
    n_true = sum(len(v) for v in truth.values())
    hit = sum(1 for s, ms in truth.items() for m in ms if (s, m) in pos)
    oracle = {s: {m for m in ms if (s, m) in pos} for s, ms in truth.items()}
    rep = {"pairs": len(cand), "true_pairs": n_true, "pair_recall": hit / max(n_true, 1),
           "f05_ceiling": macro_f05(truth, oracle),
           "reduction_ratio": 1 - len(cand) / max(1, cand["s1_idx"].nunique() * cand["rec_idx"].nunique())}
    LOG.info("Blocking report: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()})
    return rep


def main():
    from .pipeline import load_split
    from .utils import setup_logging, write_candidate_pairs
    ap = argparse.ArgumentParser(description="Run blocking only and export candidate_pairs.tsv")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="output/candidate_pairs.tsv")
    args = ap.parse_args()
    setup_logging()
    cfg = BlockingConfig()
    split = load_split(args.data_dir)  # reports blocking recall if a ground truth is present
    spaces = build_spaces(split.s1, split.rec, cfg)
    cand = attach_ids(generate_candidates(split.s1, split.rec, spaces, cfg), split.s1, split.rec)
    write_candidate_pairs(args.out, cand)
    if split.truth is not None:
        blocking_report(cand, split.truth)


if __name__ == "__main__":
    main()
