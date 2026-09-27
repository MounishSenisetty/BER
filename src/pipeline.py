"""Shared streaming driver used by train and inference so the two paths cannot drift apart.

load split -> for each country: prepare Source 1, build CountryIndex -> for each record chunk:
prepare records, query the index (blocking) -> hand (index, records, candidates) to the caller.
Peak memory is one country's Source 1 index plus one record chunk.
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .blocking import CountryIndex, RecordMats
from .config import PipelineConfig
from .normalize import country_key, prepare
from .utils import LOG, find_file, load_ground_truth, load_source, timer


@dataclass
class Split:
    s1: pd.DataFrame                       # raw Source 1 (id, name, address, country, source)
    rec: pd.DataFrame                      # raw Source 2 + Source 3
    truth: Optional[Dict[str, Set[str]]]
    gt_header: Tuple[str, str]


@dataclass
class Chunk:
    country: str
    idx: CountryIndex
    rc: pd.DataFrame                       # prepared record chunk
    cand: pd.DataFrame                     # candidates (local indices `s1`, `rec`)
    mats: RecordMats
    s1_rows: np.ndarray                    # local S1 index -> global S1 row
    rec_rows: np.ndarray                   # local record index -> global record row

    @property
    def g_s1(self) -> np.ndarray:
        return self.s1_rows[self.cand["s1"].to_numpy()]

    @property
    def g_rec(self) -> np.ndarray:
        return self.rec_rows[self.cand["rec"].to_numpy()]


def load_split(data_dir: str, s1_path: str = None, s2_path: str = None, s3_path: str = None,
               gt_path: str = None, with_truth: bool = True) -> Split:
    paths = {"s1": s1_path or find_file(data_dir, "s1"), "s2": s2_path or find_file(data_dir, "s2"),
             "s3": s3_path or find_file(data_dir, "s3")}
    for k, v in paths.items():
        if not v or not os.path.exists(v):
            raise FileNotFoundError(f"Could not find {k} file in {data_dir} (pass --{k} explicitly)")
    with timer("Loading sources"):
        s1 = load_source(paths["s1"], 1)
        rec = pd.concat([load_source(paths["s2"], 2), load_source(paths["s3"], 3)], ignore_index=True)
    truth, header = None, ("source1_entity_id", "matched_entity_ids")
    gt = gt_path or (find_file(data_dir, "gt") if with_truth else None)
    if gt and os.path.exists(gt):
        with timer("Loading ground truth"):
            truth, header = load_ground_truth(gt, s1["id"])
        n_pos = sum(len(v) for v in truth.values())
        n_single = sum(1 for v in truth.values() if not v)
        LOG.info("Ground truth: %d entities, %d singletons (%.1f%%), %d true pairs", len(truth), n_single,
                 100 * n_single / max(len(truth), 1), n_pos)
    return Split(s1, rec, truth, header)


def partition_keys(split: Split, cfg: PipelineConfig) -> Tuple[np.ndarray, np.ndarray]:
    if not cfg.blocking.partition_by_country:
        return np.full(len(split.s1), "_all", dtype=object), np.full(len(split.rec), "_all", dtype=object)
    return (split.s1["country"].map(country_key).to_numpy(dtype=object),
            split.rec["country"].map(country_key).to_numpy(dtype=object))


def iter_chunks(split: Split, cfg: PipelineConfig) -> Iterator[Chunk]:
    k1, k2 = partition_keys(split, cfg)
    sizes = pd.Series(k1).value_counts()
    rec_sizes = pd.Series(k2).value_counts()
    LOG.info("Partitions (S1 / records): %s", {c: (int(sizes.get(c, 0)), int(rec_sizes.get(c, 0)))
                                                for c in sorted(set(sizes.index) | set(rec_sizes.index))})
    orphan = sorted(set(rec_sizes.index) - set(sizes.index))
    if orphan:
        LOG.warning("records with no Source 1 partition (they get no candidates): %s",
                    {c: int(rec_sizes[c]) for c in orphan})
    for country in sizes.index:
        s1_rows = np.flatnonzero(k1 == country)
        rec_rows = np.flatnonzero(k2 == country)
        if len(rec_rows) == 0:
            continue
        with timer(f"[{country}] preparing {len(s1_rows)} S1 rows + building index"):
            s1c = prepare(split.s1.iloc[s1_rows], n_jobs=cfg.n_jobs)
            idx = CountryIndex(s1c, cfg.blocking, n_jobs=cfg.n_jobs, seed=cfg.model.seed)
            del s1c
        step = cfg.blocking.chunk_records
        for start in range(0, len(rec_rows), step):
            rr = rec_rows[start:start + step]
            with timer(f"[{country}] records {start}-{start + len(rr)} of {len(rec_rows)}: prepare + block"):
                rc = prepare(split.rec.iloc[rr], n_jobs=cfg.n_jobs)
                cand, mats = idx.query(rc)
            yield Chunk(country, idx, rc, cand, mats, s1_rows, rr)
        del idx
        gc.collect()


def owner_array(split: Split) -> np.ndarray:
    """Global record row -> global S1 row of its true owner (-1 if unmatched)."""
    s1_pos = pd.Series(np.arange(len(split.s1)), index=split.s1["id"])
    rec_pos = pd.Series(np.arange(len(split.rec)), index=split.rec["id"])
    owner = np.full(len(split.rec), -1, dtype=np.int64)
    pairs = [(s, m) for s, ms in split.truth.items() for m in ms]
    if pairs:
        p = pd.DataFrame(pairs, columns=["s", "m"])
        p = p[p["s"].isin(s1_pos.index) & p["m"].isin(rec_pos.index)]
        owner[rec_pos[p["m"]].to_numpy()] = s1_pos[p["s"]].to_numpy()
    return owner


def entity_true_counts(split: Split) -> np.ndarray:
    """|true set| per S1 row, including matches blocking never proposed."""
    return np.fromiter((len(split.truth.get(i, ())) for i in split.s1["id"]), dtype=np.float64,
                       count=len(split.s1))


def _count_rows(path: str) -> int:
    with open(path, "rb") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def auto_s1_drop(train: Split, test_dir: str, max_frac: float = 0.5) -> float:
    """Fraction of training Source 1 entities to drop so that training has the test split's
    records-per-entity ratio.

    The test split has more Source 2/3 records per Source 1 entity than train (5.75 vs 4.68 on the
    competition data) while the number of true matches per entity is the same: the extra records are
    distractors, i.e. records of businesses that are not in Source 1. Dropping entities from the
    training Source 1 turns their records into exactly that kind of distractor, so the model learns
    the test-time prior and the test-time density of look-alike competitors."""
    paths = [find_file(test_dir, k) for k in ("s1", "s2", "s3")]
    if not all(p and os.path.exists(p) for p in paths):
        return 0.0
    n1, n23 = _count_rows(paths[0]), _count_rows(paths[1]) + _count_rows(paths[2])
    if n1 == 0:
        return 0.0
    ratio_test = n23 / n1
    ratio_train = len(train.rec) / max(len(train.s1), 1)
    frac = 1.0 - ratio_train / ratio_test
    LOG.info("Records per Source 1 entity: train %.3f, test %.3f -> drop %.1f%% of training Source 1",
             ratio_train, ratio_test, 100 * max(frac, 0.0))
    return float(min(max(frac, 0.0), max_frac))


def thin_source1(split: Split, frac: float, seed: int) -> Split:
    """Remove a random `frac` of Source 1 entities; their matched records stay as distractors."""
    if frac <= 0 or split.truth is None:
        return split
    rng = np.random.default_rng(seed + 991)
    keep = rng.random(len(split.s1)) >= frac
    s1 = split.s1[keep].reset_index(drop=True)
    kept = set(s1["id"])
    truth = {k: v for k, v in split.truth.items() if k in kept}
    n_dist = sum(len(v) for k, v in split.truth.items() if k not in kept)
    LOG.info("S1 dropout: kept %d of %d Source 1 entities; %d matched records became distractors",
             len(s1), len(split.s1), n_dist)
    return Split(s1, split.rec, truth, split.gt_header)
