"""Stage 2: re-score every pair with the stage-1 probabilities of its competitors.

Stage 1 judges a pair in isolation (plus record-level context of cheap similarities). Stage 2 adds
what only becomes visible once *every* pair has a probability:

  record side  - a record has exactly one owner: this pair's rank among the record's candidates,
                 the best competing probability, the margin to it, how many strong competitors exist
  entity side  - how many strong candidates the Source 1 entity has, this pair's rank among them,
                 the entity's best probability, total expected matches (sum of p)

These aggregates need all pairs, so they are computed after the chunked pass. Per pair only the
stage-1 probability and a few raw features (RAW) are kept, which keeps memory small (~40 B/pair).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

# raw stage-1 features carried into stage 2 (cheap to keep for ~100M test pairs)
RAW = ["cheap", "cos_full_c", "cos_name_c", "cos_addr_c", "name_tset", "addr_len2", "s1_name_freq",
       "combo", "house_eq", "tok_name_wjac", "rec_source", "n_cand_rec"]


class GroupStats:
    """Per-group top-1 / top-2 / sum / counts of p, plus each pair's rank inside its group."""

    def __init__(self, g: np.ndarray, p: np.ndarray, n_groups: int):
        order = np.lexsort((-p, g))
        gs, ps = g[order], p[order]
        start = np.r_[True, gs[1:] != gs[:-1]] if len(gs) else np.zeros(0, bool)
        first = np.flatnonzero(start)
        gid = np.cumsum(start) - 1
        pos = np.arange(len(gs)) - first[gid] if len(gs) else np.zeros(0, np.int64)
        self.rank = np.empty(len(g), dtype=np.int32)
        self.rank[order] = pos + 1
        self.top1 = np.full(n_groups, np.nan, np.float32)
        self.top2 = np.full(n_groups, np.nan, np.float32)
        self.top1[gs[first]] = ps[first]
        second = first + 1
        has2 = second < len(gs)
        has2[has2] &= gs[second[has2]] == gs[first[has2]]
        self.top2[gs[first[has2]]] = ps[second[has2]]
        self.sum = np.bincount(g, weights=p, minlength=n_groups).astype(np.float32)
        self.n = np.bincount(g, minlength=n_groups).astype(np.float32)
        self.c5 = np.bincount(g, weights=(p > 0.5), minlength=n_groups).astype(np.float32)
        self.c9 = np.bincount(g, weights=(p > 0.9), minlength=n_groups).astype(np.float32)


def _side(prefix: str, st: GroupStats, g: np.ndarray, p: np.ndarray, sel: slice) -> Dict[str, np.ndarray]:
    gg, pp, rk = g[sel], p[sel], st.rank[sel]
    other = np.where(rk == 1, st.top2[gg], st.top1[gg])            # best competitor
    return {
        f"{prefix}_rank": rk.astype(np.float32),
        f"{prefix}_other": other,
        f"{prefix}_gap": pp - np.nan_to_num(other, nan=0.0),
        f"{prefix}_ratio": pp / np.maximum(st.top1[gg], 1e-6),
        f"{prefix}_sum_other": st.sum[gg] - pp,
        f"{prefix}_n": st.n[gg],
        f"{prefix}_c5_other": st.c5[gg] - (pp > 0.5),
        f"{prefix}_c9_other": st.c9[gg] - (pp > 0.9),
    }


def stage2_frame(s1: np.ndarray, rec: np.ndarray, p1: np.ndarray, raw: Dict[str, np.ndarray],
                 rec_stats: GroupStats, s1_stats: GroupStats, sel=slice(None)) -> pd.DataFrame:
    p = p1[sel].astype(np.float32)
    F = {"p1": p, "p1_logit": np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))}
    F.update(_side("rec", rec_stats, rec, p1, sel))
    F.update(_side("ent", s1_stats, s1, p1, sel))
    for k in RAW:
        F[k] = np.asarray(raw[k][sel], dtype=np.float32)
    return pd.DataFrame(F)


def predict_in_batches(ens, s1, rec, p1, raw, n_s1: int, n_rec: int, batch: int = 5_000_000) -> np.ndarray:
    """Stage-2 probabilities for every pair, assembling features batch by batch to bound memory."""
    rs = GroupStats(rec, p1, n_rec)
    ss = GroupStats(s1, p1, n_s1)
    out = np.empty(len(p1), dtype=np.float64)
    for a in range(0, len(p1), batch):
        sl = slice(a, min(a + batch, len(p1)))
        out[sl] = ens.predict(stage2_frame(s1, rec, p1, raw, rs, ss, sl))
    return out
