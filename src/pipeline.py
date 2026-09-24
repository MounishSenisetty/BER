"""Shared glue: load a split -> normalise -> block -> featurise. Used by train and inference so
the two paths cannot drift apart."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .blocking import attach_ids, build_spaces, generate_candidates
from .config import PipelineConfig
from .features import compute_features
from .normalize import prepare
from .utils import LOG, find_file, load_ground_truth, load_source, timer


@dataclass
class Split:
    s1: pd.DataFrame                       # prepared Source 1
    rec: pd.DataFrame                      # prepared Source 2 + Source 3 (column `source` in {2,3})
    truth: Optional[Dict[str, Set[str]]]
    gt_header: Tuple[str, str]


def load_split(data_dir: str, s1_path: str = None, s2_path: str = None, s3_path: str = None,
               gt_path: str = None, with_truth: bool = True) -> Split:
    paths = {"s1": s1_path or find_file(data_dir, "s1"), "s2": s2_path or find_file(data_dir, "s2"),
             "s3": s3_path or find_file(data_dir, "s3")}
    for k, v in paths.items():
        if not v or not os.path.exists(v):
            raise FileNotFoundError(f"Could not find {k} file in {data_dir} (pass --{k} explicitly)")
    s1 = load_source(paths["s1"], 1)
    rec = pd.concat([load_source(paths["s2"], 2), load_source(paths["s3"], 3)], ignore_index=True)
    clash = set(s1["id"]) & set(rec["id"])
    dup_rec = rec["id"].duplicated().sum()
    if clash or dup_rec:
        LOG.warning("id collisions: %d S1/S23, %d within S2+S3 -- output ids are ambiguous", len(clash), dup_rec)

    with timer("Normalising records"):
        s1, rec = prepare(s1), prepare(rec)

    truth, header = None, ("source1_id", "matches")
    gt = gt_path or (find_file(data_dir, "gt") if with_truth else None)
    if gt and os.path.exists(gt):
        truth, header = load_ground_truth(gt, s1["id"])
        n_pos = sum(len(v) for v in truth.values())
        n_single = sum(1 for v in truth.values() if not v)
        LOG.info("Ground truth: %d entities, %d singletons (%.1f%%), %d true pairs", len(truth), n_single,
                 100 * n_single / max(len(truth), 1), n_pos)
    return Split(s1, rec, truth, header)


def build_pairs(split: Split, cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (candidate pairs with ids, feature matrix aligned row-by-row)."""
    spaces = build_spaces(split.s1, split.rec, cfg.blocking)
    cand = generate_candidates(split.s1, split.rec, spaces, cfg.blocking)
    cand = attach_ids(cand, split.s1, split.rec)
    feats = compute_features(split.s1, split.rec, cand, spaces, n_jobs=cfg.n_jobs)
    return cand, feats


def pair_labels(cand: pd.DataFrame, truth: Dict[str, Set[str]]) -> np.ndarray:
    pos = {(s, m) for s, ms in truth.items() for m in ms}
    return np.fromiter(((s, c) in pos for s, c in zip(cand["source1_id"], cand["candidate_id"])),
                       dtype=bool, count=len(cand))


def entity_true_counts(split: Split) -> np.ndarray:
    """|true set| per S1 row (aligned with split.s1), including matches blocking never proposed."""
    return np.array([len(split.truth.get(i, ())) for i in split.s1["id"]], dtype=np.float64)
