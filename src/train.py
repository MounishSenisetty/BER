"""Train the pair classifier with entity-grouped CV and tune the macro-F0.5 decision rule.

    python -m src.train --data-dir <.../dataset/train> --model-dir models

Blocking runs over the FULL training split (so candidate density, competition between similar
entities and the context features look exactly like test). Features and labels are then built
for a stratified sample of Source 1 entities (`--train-entities`, default 120k): every
candidate pair of a sampled entity is kept, so per-entity macro F0.5 can be evaluated honestly.

Outputs (model-dir):
    model_bundle.pkl          fold boosters + feature list + tuned decision rule (used by inference)
    cv_report.json            blocking recall / ceiling, OOF logloss/AUC, macro F0.5 (tuned + nested)
    feature_importance.csv    mean gain / split importance across folds
    threshold_curve.csv       macro F0.5 for every (mode, threshold) evaluated
    oof_predictions.tsv       OOF probability per sampled candidate pair (error analysis)
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

from .config import PipelineConfig
from .features import compute_features
from .pipeline import entity_true_counts, iter_chunks, load_split, owner_array
from .postprocess import search_decision, select
from .utils import peak_memory_gb, LOG, fast_macro_f05, setup_logging, timer


def sample_entities(n_true: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Boolean mask over S1 rows: stratified (by #matches 0/1/2/3+) sample of n entities."""
    N = len(n_true)
    if n <= 0 or n >= N:
        return np.ones(N, dtype=bool)
    strata = np.minimum(n_true, 3).astype(int)
    rng = np.random.default_rng(seed)
    mask = np.zeros(N, dtype=bool)
    for s in np.unique(strata):
        rows = np.flatnonzero(strata == s)
        k = int(round(n * len(rows) / N))
        mask[rng.choice(rows, size=min(k, len(rows)), replace=False)] = True
    return mask


def entity_folds(n_true_e: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    strata = np.minimum(n_true_e, 3).astype(int)
    counts = np.bincount(strata)
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed) \
        if counts[counts > 0].min() >= n_folds else KFold(n_folds, shuffle=True, random_state=seed)
    folds = np.empty(len(strata), dtype=np.int16)
    for f, (_, va) in enumerate(splitter.split(np.zeros(len(strata)), strata)):
        folds[va] = f
    return folds


def build_training_set(split, cfg, n_true, owner, in_E):
    """Blocking over the full split + features for the sampled entities' candidate pairs."""
    K = cfg.blocking.max_candidates_per_record
    found = np.zeros(len(n_true))
    rank_hist = np.zeros(K + 1)
    n_pairs, n_recs = 0, 0
    Xs, ys, s1s, recs = [], [], [], []
    for ch in iter_chunks(split, cfg):
        gs1, grec = ch.g_s1, ch.g_rec
        y = owner[grec] == gs1
        np.add.at(found, gs1[y], 1)
        rank_hist += np.bincount(ch.cand["cheap_rank"].to_numpy()[y], minlength=K + 1)[:K + 1]
        n_pairs += len(ch.cand)
        n_recs += len(ch.rc)
        e = in_E[gs1]
        if not e.any():
            continue
        # featurise every candidate of the records that touch a sampled entity (so record-context
        # features see the full competition), then keep the sampled entities' pairs
        rec_touch = np.zeros(len(ch.rc), dtype=bool)
        rec_touch[ch.cand["rec"].to_numpy()[e]] = True
        sub = rec_touch[ch.cand["rec"].to_numpy()]
        cand = ch.cand[sub].reset_index(drop=True)
        with timer(f"  features for {len(cand)} pairs"):
            X = compute_features(ch.idx, ch.rc, cand, ch.mats, n_jobs=cfg.n_jobs)
        keep = e[sub]
        Xs.append(X[keep].reset_index(drop=True))
        ys.append(y[sub][keep])
        s1s.append(gs1[sub][keep])
        recs.append(grec[sub][keep])
        del X

    # ------------------------------------------------------------ blocking report
    total_true = n_true.sum()
    ceiling = np.where(n_true == 0, 1.0, 1.25 * found / np.maximum(0.25 * n_true + found, 1e-9))
    recall_at = {int(k): float(rank_hist[1:k + 1].sum() / max(total_true, 1)) for k in range(1, K + 1)}
    blk = {"pairs": int(n_pairs), "records": int(n_recs), "pairs_per_record": n_pairs / max(n_recs, 1),
           "true_pairs": int(total_true), "pair_recall": float(found.sum() / max(total_true, 1)),
           "f05_ceiling_all": float(ceiling.mean()), "f05_ceiling_matched": float(ceiling[n_true > 0].mean()),
           "recall_at_cap": recall_at}
    LOG.info("Blocking report: %s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in blk.items()
                                     if k != "recall_at_cap"})
    LOG.info("Pair recall by per-record cap: %s", {k: round(v, 4) for k, v in recall_at.items()})

    X = pd.concat(Xs, ignore_index=True)
    y = np.concatenate(ys)
    ps1 = np.concatenate(s1s)
    prec = np.concatenate(recs)
    del Xs
    return X, y, ps1, prec, blk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--s1"), ap.add_argument("--s2"), ap.add_argument("--s3"), ap.add_argument("--gt")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--train-entities", type=int, default=None, help="S1 entities to featurise (0 = all)")
    ap.add_argument("--max-candidates", type=int, default=None, help="candidates kept per record")
    ap.add_argument("--chunk-records", type=int, default=None)
    ap.add_argument("--feature-cache", default=None,
                    help="pickle of the training feature matrix: loaded if it exists, else written")
    ap.add_argument("--lgb-params", default=None, help='JSON overrides of the LightGBM params, e.g. \'{"num_leaves": 63}\'')
    args = ap.parse_args()
    setup_logging()
    os.makedirs(args.model_dir, exist_ok=True)

    cfg = PipelineConfig()
    if args.folds:
        cfg.model.n_folds = args.folds
    if args.train_entities is not None:
        cfg.model.train_entities = args.train_entities
    if args.max_candidates:
        cfg.blocking.max_candidates_per_record = args.max_candidates
    if args.chunk_records:
        cfg.blocking.chunk_records = args.chunk_records
    if args.lgb_params:
        cfg.model.params.update(json.loads(args.lgb_params))

    split = load_split(args.data_dir, args.s1, args.s2, args.s3, args.gt, with_truth=True)
    if split.truth is None:
        raise SystemExit("Training needs a ground-truth file (--gt)")
    n_true = entity_true_counts(split)
    owner = owner_array(split)
    in_E = sample_entities(n_true, cfg.model.train_entities, cfg.model.seed)
    LOG.info("Training entities: %d of %d (singletons %.1f%%)", in_E.sum(), len(in_E),
             100 * (n_true[in_E] == 0).mean())

    cache = args.feature_cache
    if cache and os.path.exists(cache):
        with timer(f"Loading feature cache {cache}"):
            with open(cache, "rb") as fh:
                X, y, ps1, prec, blk = pickle.load(fh)
    else:
        X, y, ps1, prec, blk = build_training_set(split, cfg, n_true, owner, in_E)
        if cache:
            with open(cache, "wb") as fh:
                pickle.dump((X, y, ps1, prec, blk), fh, protocol=4)
    LOG.info("Training pairs: %d | positives: %d (%.2f%%) | features: %d", len(y), y.sum(), 100 * y.mean(), X.shape[1])

    # ------------------------------------------------------------ entity-grouped CV
    E_rows = np.flatnonzero(in_E)
    e_local = np.full(len(in_E), -1, dtype=np.int64)
    e_local[E_rows] = np.arange(len(E_rows))
    ent = e_local[ps1]
    n_true_E = n_true[E_rows]
    ent_fold = entity_folds(n_true_E, cfg.model.n_folds, cfg.model.seed)
    pair_fold = ent_fold[ent]

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
                                  "split": b.feature_importance("split")}))

    imp = (pd.concat(imps).groupby("feature")[["gain", "split"]].mean()
           .sort_values("gain", ascending=False).reset_index())
    imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
    imp.to_csv(os.path.join(args.model_dir, "feature_importance.csv"), index=False)
    LOG.info("Top features:\n%s", imp.head(15).to_string(index=False))
    oof_ll = log_loss(y, oof, labels=[0, 1])
    oof_auc = roc_auc_score(y, oof) if 0 < y.sum() < len(y) else float("nan")
    LOG.info("OOF logloss %.5f | AUC %.5f", oof_ll, oof_auc)

    # ------------------------------------------------------------ decision rule for macro F0.5
    dc = cfg.decision
    best, curve = search_decision(ent, prec, oof, y, n_true_E, dc.modes, dc.exclusive_options, dc.grid)
    curve.to_csv(os.path.join(args.model_dir, "threshold_curve.csv"), index=False)
    sel = select(ent, prec, oof, best["mode"], best["threshold"], best["exclusive"])
    single = n_true_E == 0

    def breakdown(mask_e):
        E = len(n_true_E)
        tp = np.bincount(ent, weights=(sel & y).astype(float), minlength=E)
        npred = np.bincount(ent, weights=sel.astype(float), minlength=E)
        denom = 0.25 * n_true_E + npred
        fe = np.where(denom > 0, 1.25 * tp / np.maximum(denom, 1e-12), 1.0)
        return float(fe[mask_e].mean()) if mask_e.any() else float("nan")

    tuned = {"macro_f05": breakdown(np.ones(len(n_true_E), bool)), "singleton_f05": breakdown(single),
             "matched_f05": breakdown(~single)}
    LOG.info("OOF (rule tuned on all folds): %s", tuned)

    nested, weights = [], []
    for f in range(cfg.model.n_folds):
        in_f = pair_fold == f
        n_true_other = np.where(ent_fold != f, n_true_E, 0.0)
        b_f, _ = search_decision(ent[~in_f], prec[~in_f], oof[~in_f], y[~in_f], n_true_other,
                                 dc.modes, dc.exclusive_options, dc.grid[::2], verbose=False)
        sel_f = select(ent, prec, oof, b_f["mode"], b_f["threshold"], b_f["exclusive"])
        ents_f = ent_fold == f
        n_true_f = np.where(ents_f, n_true_E, 0.0)
        score_all = fast_macro_f05(ent[in_f], sel_f[in_f], y[in_f], n_true_f)
        # fast_macro_f05 averages over all E entities; entities outside fold f score 1 (0/0) -> rescale
        E, nf = len(n_true_E), ents_f.sum()
        nested.append((score_all * E - (E - nf)) / nf)
        weights.append(nf)
    nested_score = float(np.average(nested, weights=weights))
    LOG.info("Nested-CV macro F0.5 per fold: %s -> %.5f", np.round(nested, 5).tolist(), nested_score)

    pd.DataFrame({"source1_entity_id": split.s1["id"].to_numpy()[ps1], "candidate_entity_id": split.rec["id"].to_numpy()[prec],
                  "label": y.astype(int), "oof_prob": oof, "fold": pair_fold, "selected": sel.astype(int)}) \
        .to_csv(os.path.join(args.model_dir, "oof_predictions.tsv"), sep="\t", index=False)
    report = {"blocking": blk, "oof_logloss": oof_ll, "oof_auc": oof_auc, "decision": best,
              "oof_tuned": tuned, "nested_cv_macro_f05": nested_score, "nested_per_fold": nested,
              "best_iterations": [b.best_iteration for b in boosters], "n_features": X.shape[1],
              "train_pairs": int(len(y)), "train_entities": int(in_E.sum())}
    with open(os.path.join(args.model_dir, "cv_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    bundle = {"boosters": [b.model_to_string(num_iteration=b.best_iteration) for b in boosters],
              "features": list(X.columns), "decision": best, "config": cfg}
    with open(os.path.join(args.model_dir, "model_bundle.pkl"), "wb") as fh:
        pickle.dump(bundle, fh)
    LOG.info("Saved model bundle to %s", os.path.join(args.model_dir, "model_bundle.pkl"))
    LOG.info("Memory: %s", peak_memory_gb())


if __name__ == "__main__":
    main()
