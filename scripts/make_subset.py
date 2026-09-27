"""Build a smaller but realistic training split by sampling whole states (quick experiments on a
small machine).

Entity resolution difficulty comes from look-alike businesses in the same area, so sampling
entities at random would make the task easier than it is. This keeps a random fraction of
*states*: every Source 1 entity located there, every record whose address names one of them, and
records without a recognisable state (address-less, native-script state) when their owner is kept
(unmatched ones at the same rate). Records per entity, the distractor share and local density are
all preserved.

    python scripts/make_subset.py --train-dir <dataset/train> --out <dir>/train --frac 0.3
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.normalize import IN_STATES, US_STATES  # noqa: E402

_STATES = {**{k: v for k, v in US_STATES.items()}, **{v: v for v in US_STATES.values()},
           **{k: "in_" + v for k, v in IN_STATES.items()}, **{v: "in_" + v for v in IN_STATES.values()}}


def state_of(addr: str, country: str) -> str:
    parts = [re.sub(r"[^a-z ]+", " ", p.lower()).strip() for p in str(addr).split(",")]
    for p in reversed(parts):
        p = re.sub(r"\s+", " ", p)
        for cand in (p, p.split(" ")[-1] if p else ""):
            code = _STATES.get(cand)
            if code and (country != "India" or code.startswith("in_") or len(cand) > 2):
                return country + ":" + code
    return ""


def keep_state(st: str, frac: float, seed: int) -> bool:
    h = int(hashlib.md5(f"{seed}:{st}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--states", default=None, help="explicit comma-separated state keys, e.g. US:nc,India:in_wb")
    a = ap.parse_args()
    rd = lambda f: pd.read_csv(os.path.join(a.train_dir, f), sep="\t", dtype=str, keep_default_na=False, quoting=3)
    s1 = rd("train_source1.tsv")
    gt = rd("train_ground_truth.tsv")
    st1 = np.array([state_of(x, c) for x, c in zip(s1.business_address, s1.country)], dtype=object)
    states = sorted(set(st1) - {""})
    kept_states = set(a.states.split(",")) if a.states else {s for s in states if keep_state(s, a.frac, a.seed)}
    k1 = np.array([s in kept_states for s in st1])
    print(f"S1 with a state: {(st1 != '').mean():.3f}; states kept {len(kept_states)}/{len(states)}; S1 kept {k1.sum()}")
    s1k = s1[k1]
    kept_ids = set(s1k.entity_id)
    pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    owner = dict(zip(pairs.m, pairs.source1_entity_id))
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    kept_recs = set()
    for k in (2, 3):
        r = rd(f"train_source{k}.tsv")
        st = np.array([state_of(x, c) for x, c in zip(r.business_address, r.country)], dtype=object)
        own = r.entity_id.map(owner)
        known = st != ""
        keep = np.where(known, [s in kept_states for s in st],
                        np.where(own.notna(), own.isin(kept_ids), rng.random(len(r)) < a.frac))
        # a record kept through its state but owned by a dropped entity would be a spurious distractor
        keep &= ~(own.notna() & ~own.isin(kept_ids)).to_numpy()
        rk = r[keep]
        kept_recs |= set(rk.entity_id)
        rk.to_csv(os.path.join(a.out, f"train_source{k}.tsv"), sep="\t", index=False)
        print(f"source {k}: kept {len(rk)} of {len(r)} (state known {known.mean():.3f})")
    s1k.to_csv(os.path.join(a.out, "train_source1.tsv"), sep="\t", index=False)
    g = gt[gt.source1_entity_id.isin(kept_ids)].copy()
    g["matched_entity_ids"] = [",".join(x for x in m.split(",") if x in kept_recs) for m in g.matched_entity_ids]
    g.to_csv(os.path.join(a.out, "train_ground_truth.tsv"), sep="\t", index=False)
    n_true = g.matched_entity_ids.str.count("S")
    print(f"ground truth: {len(g)} entities, {n_true.sum()} pairs, singletons {(n_true == 0).mean():.3f}, "
          f"records/S1 {len(kept_recs) / len(g):.3f}")


if __name__ == "__main__":
    main()
