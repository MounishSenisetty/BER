"""Train the pair classifier with entity-grouped CV and tune the macro-F0.5 decision rule.

    python -m src.train --data-dir data/train --model-dir models

Outputs (model-dir):
    model_bundle.pkl          fold boosters + feature list + tuned decision rule (used by inference)
    cv_report.json            blocking recall, OOF logloss/AUC, macro F0.5 (tuned + nested-honest)
    feature_importance.csv    mean gain / split importance across folds
    threshold_curve.csv       macro F0.5 for every (mode, exclusive, threshold) evaluated
    oof_predictions.tsv       OOF probability per candidate pair (for error analysis)
    train_candidate_pairs.tsv blocking output on train
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from .blocking import blocking_report
from .config import PipelineConfig
from .pipeline import build_pairs, entity_true_counts, load_split, pair_labels
from .postprocess import search_decision, select, to_predictions
from .utils import LOG, macro_f05, macro_f05_breakdown, setup_logging, timer, write_candidate_pairs


def make_entity_folds(s1_ids, truth, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per Source 1 entity. All candidate pairs of an entity share its fold, so an
    entity's match set is never split between train and validation. Stratified on the number
    of true matches (0 / 1 / 2 / 3+) so every fold has the same singleton rate."""
    strata = np.array([min(len(truth.get(i, ())), 3) for i in s1_ids])
    folds = np.empty(len(s1_ids), dtype=np.int16)
    counts = np.bincount(strata)
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed) \
        if counts[counts > 0].min() >= n_folds else KFold(n_folds, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(splitter.split(np.zeros(len(s1_ids)), strata)):
        folds[va] = f
    return folds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--s1"), ap.add_argument("--s2"), ap.add_argument("--s3"), ap.add_argument("--gt")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--no-lsh", action="store_true")
    args = ap.parse_args()
    setup_logging()
    os.makedirs(args.model_dir, exist_ok=True)

    cfg = PipelineConfig()
    if args.folds:
        cfg.model.n_folds = args.folds
    if args.no_lsh:
        cfg.blocking.use_lsh = False

    split = load_split(args.data_dir, args.s1, args.s2, args.s3, args.gt, with_truth=True)
    if split.truth is None:
        raise SystemExit("Training needs a ground-truth file (--gt)")
    s1_ids = split.s1["id"].tolist()
    truth = {i: split.truth.get(i, set()) for i in s1_ids}   # evaluate on S1 rows we actually have

    cand, X = build_pairs(split, cfg)
    write_candidate_pairs(os.path.join(args.model_dir, "train_candidate_pairs.tsv"), cand)
    blk = blocking_report(cand, truth)
    y = pair_labels(cand, truth)
    LOG.info("Pairs: %d | positives: %d (%.2f%%) | features: %d", len(y), y.sum(), 100 * y.mean(), X.shape[1])

    # -------------------------------------------------- entity-grouped cross-validation
    ent_fold = make_entity_folds(s1_ids, truth, cfg.model.n_folds, cfg.model.seed)
    pair_fold = ent_fold[cand["s1_idx"].to_numpy()]
    oof = np.zeros(len(y), dtype=np.float64)
    boosters, imps = [], []
    params = dict(cfg.model.params, seed=cfg.model.seed, num_threads=os.cpu_count() or 1)
    for f in range(cfg.model.n_folds):
        tr, va = pair_fold != f, pair_fold == f
        with timer(f"Fold {f}: train {tr.sum()} / valid {va.sum()} pairs"):
            dtr = lgb.Dataset(X[tr], y[tr], free_raw_data=True)
            dva = lgb.Dataset(X[va], y[va], reference=dtr)
            b = lgb.train(params, dtr, cfg.model.num_boost_round, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(cfg.model.early_stopping_rounds, verbose=False),
                                     lgb.log_evaluation(0)])
            oof[va] = b.predict(X[va], num_iteration=b.best_iteration)
            LOG.info("  best_iter=%d  valid logloss=%.5f", b.best_iteration, log_loss(y[va], oof[va], labels=[0, 1]))
        boosters.append(b)
        imps.append(pd.DataFrame({"feature": X.columns, "gain": b.feature_importance("gain"),
                                  "split": b.feature_importance("split"), "fold": f}))

    imp = (pd.concat(imps).groupby("feature")[["gain", "split"]].mean()
           .sort_values("gain", ascending=False).reset_index())
    imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
    imp.to_csv(os.path.join(args.model_dir, "feature_importance.csv"), index=False)
    LOG.info("Top features:\n%s", imp.head(15).to_string(index=False))
    oof_ll = log_loss(y, oof, labels=[0, 1])
    oof_auc = roc_auc_score(y, oof) if 0 < y.sum() < len(y) else float("nan")
    LOG.info("OOF logloss %.5f | AUC %.5f", oof_ll, oof_auc)

    # -------------------------------------------------- decision-rule search for macro F0.5
    ent_idx = cand["s1_idx"].to_numpy()
    rec_idx = cand["rec_idx"].to_numpy()
    n_true = entity_true_counts(split)
    dc = cfg.decision
    best, curve = search_decision(ent_idx, rec_idx, oof, y, n_true, dc.modes, dc.exclusive_options, dc.grid)
    curve.to_csv(os.path.join(args.model_dir, "threshold_curve.csv"), index=False)

    sel = select(ent_idx, rec_idx, oof, best["mode"], best["threshold"], best["exclusive"])
    pred = to_predictions(cand, sel, oof)
    tuned = macro_f05_breakdown(truth, pred)
    LOG.info("OOF (rule tuned on all folds): %s", tuned)

    # nested estimate: rule tuned on the other folds' OOF, applied to the held-out fold
    nested_scores, weights = [], []
    for f in range(cfg.model.n_folds):
        in_f = pair_fold == f
        ents_f = [i for i, fo in zip(s1_ids, ent_fold) if fo == f]
        ent_out = ent_fold != f
        # rows of other folds only; entities of fold f are excluded from the objective
        other = ~in_f
        n_true_other = np.where(ent_out, n_true, 0.0)
        b_f, _ = search_decision(ent_idx[other], rec_idx[other], oof[other], y[other], n_true_other,
                                 dc.modes, dc.exclusive_options, dc.grid[::2], verbose=False) if other.any() else (best, None)
        sel_f = select(ent_idx, rec_idx, oof, b_f["mode"], b_f["threshold"], b_f["exclusive"]) & in_f
        pred_f = to_predictions(cand, sel_f, oof)
        nested_scores.append(macro_f05({i: truth[i] for i in ents_f}, pred_f))
        weights.append(len(ents_f))
    nested = float(np.average(nested_scores, weights=weights))
    LOG.info("Nested-CV macro F0.5 per fold: %s -> %.5f", np.round(nested_scores, 5).tolist(), nested)

    pd.DataFrame({"source1_id": cand["source1_id"], "candidate_id": cand["candidate_id"],
                  "candidate_source": cand["candidate_source"], "label": y.astype(int),
                  "oof_prob": oof, "fold": pair_fold, "selected": sel.astype(int)}) \
        .to_csv(os.path.join(args.model_dir, "oof_predictions.tsv"), sep="\t", index=False)

    report = {"blocking": blk, "oof_logloss": oof_ll, "oof_auc": oof_auc, "decision": best,
              "oof_tuned": tuned, "nested_cv_macro_f05": nested, "nested_per_fold": nested_scores,
              "best_iterations": [b.best_iteration for b in boosters], "n_features": X.shape[1]}
    with open(os.path.join(args.model_dir, "cv_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)

    bundle = {"boosters": [b.model_to_string(num_iteration=b.best_iteration) for b in boosters],
              "features": list(X.columns), "decision": best, "gt_header": split.gt_header,
              "config": cfg}
    with open(os.path.join(args.model_dir, "model_bundle.pkl"), "wb") as fh:
        pickle.dump(bundle, fh)
    LOG.info("Saved model bundle to %s", os.path.join(args.model_dir, "model_bundle.pkl"))


if __name__ == "__main__":
    main()
