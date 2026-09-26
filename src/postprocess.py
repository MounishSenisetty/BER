"""Turning pair probabilities into per-entity match sets that maximise macro F0.5.

Two structural facts drive the decision layer:

1. Exclusivity. Source 1 is deduplicated, so a Source 2/3 record belongs to at most ONE S1
   entity. Keeping only the arg-max S1 for each record removes a whole class of false
   positives (chain stores, neighbouring businesses) at almost no recall cost.

2. Macro F0.5 is computed per entity. For an entity with candidate probabilities p_1>=p_2>=...
   predicting the top-k set gives (ratio-of-expectations approximation)

        E[F0.5 | k]  ~=  1.25 * sum_{i<=k} p_i / (0.25 * sum_i p_i + k)          (k >= 1)
        E[F0.5 | 0]   =  prod_i (1 - p_i)          (= P(entity is a singleton))

   'expected_f' mode picks k* = argmax_k E[F0.5 | k] per entity; this adapts the effective
   threshold to each entity (a lone 0.6 candidate is kept, a 0.6 candidate next to a 0.99 one is
   usually dropped). 'threshold' mode is the classic global cut. Both are tuned on OOF.

3. Entity state. Pairs are scored independently, but a second candidate of an entity that already
   has a confident match is far more likely to be real than the best candidate of an entity that
   may be a singleton. 'two_floor' mode cuts an entity's top candidate at `threshold` and every
   other candidate at `threshold2`, so the two populations get their own operating points.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import BETA2, LOG, fast_macro_f05


def rec_argmax_mask(rec_idx: np.ndarray, p: np.ndarray) -> np.ndarray:
    """True for the single highest-probability S1 candidate of every record."""
    order = np.lexsort((-p, rec_idx))
    first = np.ones(len(order), dtype=bool)
    first[1:] = rec_idx[order][1:] != rec_idx[order][:-1]
    mask = np.zeros(len(p), dtype=bool)
    mask[order[first]] = True
    return mask


def select(ent_idx: np.ndarray, rec_idx: np.ndarray, p: np.ndarray, mode: str, threshold: float,
           exclusive: bool, _argmax: np.ndarray = None, threshold2: float = None,
           _top1: np.ndarray = None) -> np.ndarray:
    """Boolean mask of pairs predicted as matches."""
    elig = np.ones(len(p), dtype=bool)
    if exclusive:
        elig = rec_argmax_mask(rec_idx, p) if _argmax is None else _argmax
    if mode == "threshold":
        return elig & (p >= threshold)
    if mode == "two_floor":
        top1 = rec_argmax_mask(ent_idx, p) if _top1 is None else _top1   # best candidate per entity
        return elig & (p >= np.where(top1, threshold, threshold2))
    if mode != "expected_f":
        raise ValueError(mode)

    idx = np.flatnonzero(elig)
    e, pe = ent_idx[idx], p[idx]
    E = int(ent_idx.max()) + 1 if len(ent_idx) else 0
    S = np.bincount(e, weights=pe, minlength=E)                                   # E|T|
    logE0 = np.bincount(e, weights=np.log1p(-np.clip(pe, 0, 1 - 1e-7)), minlength=E)

    keep = pe >= threshold
    idx, e, pe = idx[keep], e[keep], pe[keep]
    order = np.lexsort((-pe, e))
    idx, e, pe = idx[order], e[order], pe[order]
    if len(idx) == 0:
        return np.zeros(len(p), dtype=bool)
    new_grp = np.r_[True, e[1:] != e[:-1]]
    grp_start = np.flatnonzero(new_grp)
    grp_id = np.cumsum(new_grp) - 1
    k = np.arange(len(e)) - grp_start[grp_id] + 1
    cs = np.cumsum(pe)
    cum = cs - np.r_[0.0, cs][grp_start][grp_id]
    Ek = (1 + BETA2) * cum / (BETA2 * S[e] + k)
    best = np.maximum.reduceat(Ek, grp_start)
    # k* = first k attaining the max (Ek is unimodal in practice)
    is_best = Ek >= best[grp_id] - 1e-12
    kstar = np.full(len(grp_start), np.iinfo(np.int64).max)
    np.minimum.at(kstar, grp_id[is_best], k[is_best])
    choose = np.where(best > np.exp(logE0[e[grp_start]]), kstar, 0)
    out = np.zeros(len(p), dtype=bool)
    out[idx[k <= choose[grp_id]]] = True
    return out


def search_decision(ent_idx: np.ndarray, rec_idx: np.ndarray, p: np.ndarray, label: np.ndarray,
                    n_true: np.ndarray, modes, exclusive_options, grid,
                    verbose: bool = True) -> Tuple[Dict, pd.DataFrame]:
    """Grid-search (mode, exclusive, threshold) for the best macro F0.5 over ALL entities.
    `n_true` must include true matches that blocking missed, so recall loss is priced in."""
    rows = []
    argmax = rec_argmax_mask(rec_idx, p)
    top1 = rec_argmax_mask(ent_idx, p)
    grid2 = grid[::2]                         # 2-D search: coarser steps keep it to ~2.4k evaluations
    for exclusive in exclusive_options:
        for mode in modes:
            pairs = [(t1, t2) for t1 in grid2 for t2 in grid2] if mode == "two_floor" else [(t, np.nan) for t in grid]
            for t, t2 in pairs:
                sel = select(ent_idx, rec_idx, p, mode, t, exclusive, _argmax=argmax if exclusive else None,
                             threshold2=t2, _top1=top1)
                rows.append({"mode": mode, "exclusive": exclusive, "threshold": float(t), "threshold2": float(t2),
                             "macro_f05": fast_macro_f05(ent_idx, sel, label, n_true),
                             "n_pred": int(sel.sum())})
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["macro_f05"].idxmax()].to_dict()
    # prefer thresholds in the middle of the plateau (robust to calibration shift on test): the
    # plateau point closest to the plateau's median (per axis for the 2-D mode)
    same = curve[(curve["mode"] == best["mode"]) & (curve["exclusive"] == best["exclusive"])]
    plateau = same[same["macro_f05"] >= best["macro_f05"] - 5e-4]
    dist = (plateau["threshold"] - plateau["threshold"].median()).abs()
    if best["mode"] == "two_floor":
        dist = dist + (plateau["threshold2"] - plateau["threshold2"].median()).abs()
    at = plateau.loc[dist.idxmin()]
    best["threshold"], best["threshold2"] = float(at["threshold"]), float(at["threshold2"])
    best["macro_f05"], best["n_pred"] = float(at["macro_f05"]), int(at["n_pred"])
    if not verbose:
        return best, curve
    LOG.info("Best decision rule: %s", best)
    for (m, ex), g in curve.groupby(["mode", "exclusive"]):
        r = g.loc[g["macro_f05"].idxmax()]
        LOG.info("  mode=%-10s exclusive=%-5s best t=%.2f t2=%.2f -> macro F0.5 %.5f", m, ex, r["threshold"],
                 r["threshold2"], r["macro_f05"])
    return best, curve


def to_predictions(cand: pd.DataFrame, selected: np.ndarray, p: np.ndarray) -> Dict[str, List[str]]:
    """{source1_id: [candidate ids ordered by probability]} for the selected pairs."""
    d = pd.DataFrame({"s": cand["source1_id"].to_numpy()[selected],
                      "c": cand["candidate_id"].to_numpy()[selected], "p": p[selected]})
    d = d.sort_values(["s", "p"], ascending=[True, False])
    return {s: list(dict.fromkeys(g["c"])) for s, g in d.groupby("s", sort=False)}
