"""End-to-end inference on a split without labels.

    python -m src.inference --data-dir data/test --model-dir models --out-dir output

Writes:
    <out-dir>/candidate_pairs.tsv     blocking output (source1_id, candidate_id, candidate_source)
    <out-dir>/matching_results.tsv    one row per Source 1 entity -> comma-separated matches
    <out-dir>/pair_scores.tsv         (optional, --dump-scores) probability for every candidate pair

If the split contains a ground-truth file (or --gt is given) the macro F0.5 is also reported,
which makes this script double as a hold-out evaluator.
"""
from __future__ import annotations

import argparse
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from .pipeline import build_pairs, load_split
from .postprocess import select, to_predictions
from .utils import (LOG, macro_f05_breakdown, setup_logging, timer, write_candidate_pairs,
                    write_matching_results)


def predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    X = X[bundle["features"]]
    p = np.zeros(len(X), dtype=np.float64)
    for s in bundle["boosters"]:
        p += lgb.Booster(model_str=s).predict(X)
    return p / max(len(bundle["boosters"]), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/test")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--s1"), ap.add_argument("--s2"), ap.add_argument("--s3"), ap.add_argument("--gt")
    ap.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    ap.add_argument("--dump-scores", action="store_true")
    args = ap.parse_args()
    setup_logging()

    with open(os.path.join(args.model_dir, "model_bundle.pkl"), "rb") as fh:
        bundle = pickle.load(fh)
    cfg = bundle["config"]
    rule = dict(bundle["decision"])
    if args.threshold is not None:
        rule["threshold"] = args.threshold
    LOG.info("Decision rule: mode=%s exclusive=%s threshold=%.3f", rule["mode"], rule["exclusive"], rule["threshold"])

    split = load_split(args.data_dir, args.s1, args.s2, args.s3, args.gt, with_truth=args.gt is not None)
    cand, X = build_pairs(split, cfg)
    write_candidate_pairs(os.path.join(args.out_dir, "candidate_pairs.tsv"), cand)

    with timer("Scoring pairs"):
        p = predict_proba(bundle, X)
    sel = select(cand["s1_idx"].to_numpy(), cand["rec_idx"].to_numpy(), p,
                 rule["mode"], rule["threshold"], bool(rule["exclusive"]))
    pred = to_predictions(cand, sel, p)
    s1_ids = split.s1["id"].tolist()
    write_matching_results(os.path.join(args.out_dir, "matching_results.tsv"), s1_ids, pred,
                           header=tuple(bundle.get("gt_header", ("source1_id", "matches"))))
    LOG.info("Predicted %d matches for %d / %d Source 1 entities", int(sel.sum()), len(pred), len(s1_ids))

    if args.dump_scores:
        cand.assign(prob=p, selected=sel.astype(int))[
            ["source1_id", "candidate_id", "candidate_source", "prob", "selected"]
        ].to_csv(os.path.join(args.out_dir, "pair_scores.tsv"), sep="\t", index=False)

    if split.truth is not None:
        truth = {i: split.truth.get(i, set()) for i in s1_ids}
        from .blocking import blocking_report
        blocking_report(cand, truth)
        LOG.info("Hold-out evaluation: %s", macro_f05_breakdown(truth, pred))


if __name__ == "__main__":
    main()
