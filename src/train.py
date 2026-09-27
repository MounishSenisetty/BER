"""Train the pair classifier with entity-grouped CV and tune the macro-F0.5 decision rule.

    python -m src.train --data-dir <.../dataset/train> --model-dir models

Blocking runs over the FULL training split (so candidate density, competition between similar
entities and the context features look exactly like test). Features and labels are then built
for a stratified sample of Source 1 entities (`--train-entities`, default 120k): every
candidate pair of a sampled entity is kept, so per-entity macro F0.5 can be evaluated honestly.

Outputs (model-dir):
    model_bundle.pkl          fold models + feature list + tuned decision rule (used by inference)
    cv_report.json            blocking recall / ceiling, OOF logloss/AUC, macro F0.5 (tuned + nested)
    feature_importance.csv    mean gain / split importance across folds
    threshold_curve.csv       macro F0.5 for every (mode, threshold) evaluated
    oof_predictions.tsv       OOF probability per sampled candidate pair (error analysis)
    blocking_misses.tsv       sample of true pairs that candidate generation missed (with reason)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from .config import PipelineConfig
from .features import compute_features
from .model import Ensemble, resolve_backend, train_fold
from .stage2 import RAW, GroupStats, stage2_frame
from .cross_encoder import CEConfig, assemble, ce_features, config_dict, encode_rows, score_pairs, torch_device, train_model
from .diagnostics import loss_report
from .lexicon import learn_from_split
from .normalize import set_lexicon
from .pipeline import restrict_countries, auto_s1_drop, entity_true_counts, iter_chunks, load_split, owner_array, thin_source1
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


def train_cross_encoder(split, ce_cfg, all_s1, all_rec, all_p, n_tr, y, pair_fold, rf, n_jobs):
    """Train two cross-encoders on disjoint halves of the entity folds (fold % 2) and score every
    stage-2 pair with p1 >= min_p out of fold. A training pair is scored by the model of the other
    half; a competitor pair by the model that never saw its record's sampled entity (both models
    averaged when the record touches both halves, and at test time)."""
    rng = np.random.default_rng(ce_cfg.seed)
    grp = (pair_fold % 2).astype(np.int8)
    p_tr = all_p[:n_tr]
    pick = (y == 1) | (p_tr >= ce_cfg.train_min_p) | (rng.random(n_tr) < ce_cfg.easy_neg_frac)
    s1_tr, rec_tr = all_s1[:n_tr], all_rec[:n_tr]
    u1, t1 = encode_rows(split.s1, s1_tr[pick], ce_cfg, n_jobs)
    u2, t2 = encode_rows(split.rec, rec_tr[pick], ce_cfg, n_jobs)
    blobs = []
    for g in (0, 1):
        tr = np.flatnonzero(pick & (grp == g))
        if len(tr) > ce_cfg.max_train_pairs:
            tr = np.sort(rng.choice(tr, size=ce_cfg.max_train_pairs, replace=False))
        va = np.flatnonzero(pick & (grp != g) & (p_tr >= ce_cfg.min_p))
        if len(va) > 200_000:
            va = np.sort(rng.choice(va, size=200_000, replace=False))
        Xg = assemble(t1, np.searchsorted(u1, s1_tr[tr]), t2, np.searchsorted(u2, rec_tr[tr]))
        Xv = assemble(t1, np.searchsorted(u1, s1_tr[va]), t2, np.searchsorted(u2, rec_tr[va]))
        LOG.info("Cross-encoder %d: %d training pairs (%.1f%% positive)", g, len(tr), 100 * y[tr].mean())
        blobs.append(train_model(Xg, y[tr], ce_cfg, tag=f"[{g}]", X_val=Xv, y_val=y[va]))
        del Xg, Xv
    del t1, t2
    sub = np.flatnonzero(all_p >= ce_cfg.min_p)
    use = np.full(len(all_p), 2, dtype=np.int8)
    use[:n_tr] = 1 - grp
    rfc = rf[all_rec[n_tr:]]
    use[n_tr:] = np.where(rfc >= 0, 1 - (rfc % 2), 2)
    logit = score_pairs(blobs, use[sub], split.s1, split.rec, all_s1[sub], all_rec[sub], ce_cfg, n_jobs)
    feats = ce_features(logit, all_p[sub], all_s1[sub], all_rec[sub], len(split.s1), len(split.rec))
    return blobs, (sub, feats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--s1"), ap.add_argument("--s2"), ap.add_argument("--s3"), ap.add_argument("--gt")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--train-entities", type=int, default=None, help="S1 entities to featurise (0 = all)")
    ap.add_argument("--max-candidates", type=int, default=None, help="candidates kept per record")
    ap.add_argument("--chunk-records", type=int, default=None)
    ap.add_argument("--backend", choices=["auto", "xgboost", "lightgbm"], default=None)
    ap.add_argument("--cache-dir", default="/tmp/ber_stage2_cache",
                    help="scratch space for competitor-pair features (stage 2), ~10 GB on the full data")
    ap.add_argument("--no-stage2", action="store_true")
    ap.add_argument("--test-dir", default=None,
                    help="test split directory (default: ../test next to --data-dir); only its row counts are "
                         "read, to size the S1 dropout")
    ap.add_argument("--s1-drop", type=float, default=None,
                    help="fraction of training Source 1 entities to drop so their records become distractors "
                         "(default: auto from the test split's records-per-entity ratio; 0 = off)")
    ap.add_argument("--no-lexicon", action="store_true", help="do not learn the transliteration lexicon")
    ap.add_argument("--countries", default=None, help="comma-separated country labels to train on (quick experiments)")
    ap.add_argument("--cross-encoder", choices=["auto", "on", "off"], default="auto",
                    help="char-level transformer cross-encoder as a stage-2 feature (auto = only with a CUDA GPU)")
    ap.add_argument("--ce-epochs", type=float, default=None)
    ap.add_argument("--ce-max-train-pairs", type=int, default=None)
    ap.add_argument("--ce-small", action="store_true", help="tiny cross-encoder (CPU smoke tests)")
    ap.add_argument("--ce-backbone", default="",
                    help="fine-tune a pretrained Hugging Face model as the cross-encoder instead of the from-scratch "
                         "character model, e.g. microsoft/deberta-v3-small or intfloat/multilingual-e5-small "
                         "(both MIT; a local path works offline)")
    ap.add_argument("--ce-max-len", type=int, default=None, help="token budget per pair for --ce-backbone (default 96)")
    ap.add_argument("--final", choices=["auto", "stage1", "stage2", "stage2+ce"], default="auto",
                    help="model variant to ship (auto = best nested-CV macro F0.5)")
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
    if args.backend:
        cfg.model.backend = args.backend

    split = load_split(args.data_dir, args.s1, args.s2, args.s3, args.gt, with_truth=True)
    if split.truth is None:
        raise SystemExit("Training needs a ground-truth file (--gt)")
    if args.countries:
        keep_c = {c.strip().lower() for c in args.countries.split(",")}
        split = restrict_countries(split, keep_c)
    lexicon = {}
    if not args.no_lexicon:
        with timer("Learning the transliteration lexicon"):
            lexicon = learn_from_split(split, n_jobs=cfg.n_jobs, seed=cfg.model.seed)
        set_lexicon(lexicon)
    s1_drop = args.s1_drop
    if s1_drop is None:
        test_dir = args.test_dir or os.path.join(os.path.dirname(os.path.abspath(args.data_dir.rstrip("/"))), "test")
        s1_drop = auto_s1_drop(split, test_dir) if os.path.isdir(test_dir) else 0.0
    split = thin_source1(split, s1_drop, cfg.model.seed)
    n_true = entity_true_counts(split)
    owner = owner_array(split)
    in_E = sample_entities(n_true, cfg.model.train_entities, cfg.model.seed)
    LOG.info("Training entities: %d of %d (singletons %.1f%%)", in_E.sum(), len(in_E),
             100 * (n_true[in_E] == 0).mean())

    E_rows = np.flatnonzero(in_E)
    e_local = np.full(len(in_E), -1, dtype=np.int64)
    e_local[E_rows] = np.arange(len(E_rows))
    n_true_E = n_true[E_rows]
    ent_fold = entity_folds(n_true_E, cfg.model.n_folds, cfg.model.seed)
    fold_of_s1 = np.full(len(in_E), -1, dtype=np.int8)
    fold_of_s1[E_rows] = ent_fold
    # fold(s) of the sampled entities each record competes for -> which stage-1 model may score it
    rec_fmin = np.full(len(split.rec), 127, dtype=np.int8)
    rec_fmax = np.full(len(split.rec), -1, dtype=np.int8)
    use_s2 = not args.no_stage2
    ce_cfg = CEConfig(seed=cfg.model.seed)
    if args.ce_backbone:                   # fine-tuning recipe for a pretrained backbone
        ce_cfg.backbone = args.ce_backbone
        ce_cfg.lr, ce_cfg.batch_size, ce_cfg.epochs, ce_cfg.warmup_steps = 3e-5, 64, 1.0, 300
        ce_cfg.max_train_pairs = 600_000
        if args.ce_max_len:
            ce_cfg.max_len = args.ce_max_len
    if args.ce_epochs:
        ce_cfg.epochs = args.ce_epochs
    if args.ce_max_train_pairs:
        ce_cfg.max_train_pairs = args.ce_max_train_pairs
    if args.ce_small:
        ce_cfg.d_model, ce_cfg.n_heads, ce_cfg.n_layers, ce_cfg.ff = 64, 4, 2, 128
        ce_cfg.name_len, ce_cfg.addr_len, ce_cfg.warmup_steps = 32, 48, 50
    dev = torch_device()
    use_ce = use_s2 and (args.cross_encoder == "on" and dev != "none" or args.cross_encoder == "auto" and dev == "cuda")
    LOG.info("Cross-encoder: %s (torch device: %s, backbone: %s)", "on" if use_ce else "off", dev,
             ce_cfg.backbone or "char transformer from scratch")
    if use_s2:
        shutil.rmtree(args.cache_dir, ignore_errors=True)
        os.makedirs(args.cache_dir, exist_ok=True)
    n_cache = 0

    K = cfg.blocking.max_candidates_per_record
    found = np.zeros(len(n_true))
    found_union = 0
    rng = np.random.default_rng(cfg.model.seed)
    misses = []
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
        # diagnostics: were misses never generated, or generated and then capped away?
        u_rec, u_s1 = ch.idx.last_union
        yu = owner[ch.rec_rows[u_rec]] == ch.s1_rows[u_s1]
        found_union += int(yu.sum())
        own = owner[ch.rec_rows]
        hit = np.zeros(len(ch.rc), bool)
        hit[ch.cand["rec"].to_numpy()[y]] = True
        hit_u = np.zeros(len(ch.rc), bool)
        hit_u[u_rec[yu]] = True
        miss = np.flatnonzero((own >= 0) & ~hit)
        if len(miss):
            pick = rng.choice(miss, size=min(150, len(miss)), replace=False)
            misses += [(ch.rec_rows[i], own[i], "capped" if hit_u[i] else "not_generated") for i in pick]
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
        gr_e, f_e = grec[sub][keep], fold_of_s1[gs1[sub][keep]]
        np.minimum.at(rec_fmin, gr_e, f_e)
        np.maximum.at(rec_fmax, gr_e, f_e)
        if use_s2 and (~keep).any():     # competitor pairs (other entities) -> disk, scored after stage 1
            np.save(os.path.join(args.cache_dir, f"X_{n_cache}.npy"), X[~keep].to_numpy(np.float16))
            np.save(os.path.join(args.cache_dir, f"ids_{n_cache}.npy"),
                    np.stack([gs1[sub][~keep], grec[sub][~keep]]).astype(np.int64))
            n_cache += 1
        del X

    # ------------------------------------------------------------ blocking report
    total_true = n_true.sum()
    ceiling = np.where(n_true == 0, 1.0, 1.25 * found / np.maximum(0.25 * n_true + found, 1e-9))
    recall_at = {int(k): float(rank_hist[1:k + 1].sum() / max(total_true, 1)) for k in range(1, K + 1)}
    blk = {"pairs": int(n_pairs), "records": int(n_recs), "pairs_per_record": n_pairs / max(n_recs, 1),
           "true_pairs": int(total_true), "pair_recall": float(found.sum() / max(total_true, 1)),
           "pair_recall_before_cap": float(found_union / max(total_true, 1)),
           "f05_ceiling_all": float(ceiling.mean()), "f05_ceiling_matched": float(ceiling[n_true > 0].mean()),
           "recall_at_cap": recall_at}
    LOG.info("Blocking report: %s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in blk.items()
                                     if k != "recall_at_cap"})
    LOG.info("Pair recall by per-record cap: %s", {k: round(v, 4) for k, v in recall_at.items()})
    if misses:
        mr, ms, why = map(np.array, zip(*misses))
        S1, R = split.s1, split.rec
        md = pd.DataFrame({"reason": why, "country": S1["country"].to_numpy()[ms],
                           "s1_name": S1["name"].to_numpy()[ms], "s1_address": S1["address"].to_numpy()[ms],
                           "rec_id": R["id"].to_numpy()[mr], "rec_name": R["name"].to_numpy()[mr],
                           "rec_address": R["address"].to_numpy()[mr]})
        md.to_csv(os.path.join(args.model_dir, "blocking_misses.tsv"), sep="\t", index=False)
        LOG.info("Sampled blocking misses -> blocking_misses.tsv: %s", md["reason"].value_counts().to_dict())

    X = pd.concat(Xs, ignore_index=True)
    y = np.concatenate(ys)
    ps1 = np.concatenate(s1s)
    prec = np.concatenate(recs)
    del Xs
    LOG.info("Training pairs: %d | positives: %d (%.2f%%) | features: %d", len(y), y.sum(), 100 * y.mean(), X.shape[1])

    # ------------------------------------------------------------ entity-grouped CV
    ent = e_local[ps1]
    pair_fold = ent_fold[ent]

    oof = np.zeros(len(y), dtype=np.float64)
    models, imps, best_iters = [], [], []
    backend = resolve_backend(cfg.model.backend)
    for f in range(cfg.model.n_folds):
        tr, va = pair_fold != f, pair_fold == f
        with timer(f"Fold {f}: train {tr.sum()} / valid {va.sum()} pairs"):
            m, oof[va], imp_f = train_fold(backend, cfg.model, X[tr], y[tr], X[va], y[va])
            LOG.info("  best_iter=%d  valid logloss=%.5f", m[2], log_loss(y[va], oof[va], labels=[0, 1]))
        models.append(m)
        best_iters.append(int(m[2]))
        imps.append(imp_f)

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
    single = n_true_E == 0

    def evaluate(prob, tag):
        best, curve = search_decision(ent, prec, prob, y, n_true_E, dc.modes, dc.exclusive_options, dc.grid)
        sel = select(ent, prec, prob, best["mode"], best["threshold"], best["exclusive"])
        E = len(n_true_E)
        tp = np.bincount(ent, weights=(sel & y).astype(float), minlength=E)
        npred = np.bincount(ent, weights=sel.astype(float), minlength=E)
        denom = 0.25 * n_true_E + npred
        fe = np.where(denom > 0, 1.25 * tp / np.maximum(denom, 1e-12), 1.0)
        tuned = {"macro_f05": float(fe.mean()), "singleton_f05": float(fe[single].mean()) if single.any() else float("nan"),
                 "matched_f05": float(fe[~single].mean())}
        nested, weights = [], []
        for f in range(cfg.model.n_folds):
            in_f = pair_fold == f
            n_true_other = np.where(ent_fold != f, n_true_E, 0.0)
            b_f, _ = search_decision(ent[~in_f], prec[~in_f], prob[~in_f], y[~in_f], n_true_other,
                                     dc.modes, dc.exclusive_options, dc.grid[::2], verbose=False)
            sel_f = select(ent, prec, prob, b_f["mode"], b_f["threshold"], b_f["exclusive"])
            ents_f = ent_fold == f
            n_true_f = np.where(ents_f, n_true_E, 0.0)
            score_all = fast_macro_f05(ent[in_f], sel_f[in_f], y[in_f], n_true_f)
            nf = ents_f.sum()   # entities outside fold f score 1 (0/0) in fast_macro_f05 -> rescale
            nested.append((score_all * E - (E - nf)) / nf)
            weights.append(nf)
        nested_score = float(np.average(nested, weights=weights))
        LOG.info("[%s] OOF (rule tuned on all folds): %s", tag, tuned)
        LOG.info("[%s] Nested-CV macro F0.5 per fold: %s -> %.5f", tag, np.round(nested, 5).tolist(), nested_score)
        return {"best": best, "curve": curve, "sel": sel, "tuned": tuned, "nested": nested_score,
                "nested_per_fold": nested}

    feats_all = list(X.columns)
    ev1 = evaluate(oof, "stage1")
    final, final_prob, s2_models, s2_features, ev2 = ev1, oof, None, None, None
    variant_scores = {"stage1": ev1["nested"]}
    cands = {"stage1": (ev1, oof, None, None, False)}
    s2_uses_ce, ce_blobs = False, None

    # ------------------------------------------------------------ stage 2 (competitor-aware re-scoring)
    if use_s2:
        import gc
        feats = list(X.columns)
        raw_E = {k: X[k].to_numpy(np.float16) for k in RAW}      # float16, exactly as stored at inference
        n_features = X.shape[1]
        del X
        gc.collect()
        ens_all = Ensemble(models)
        folds_ens = [Ensemble([m]) for m in models]
        rf = np.where(rec_fmin == rec_fmax, rec_fmin, -2)      # -2: record spans several folds
        c_s1, c_rec, c_p, c_raw = [], [], [], {k: [] for k in RAW}
        with timer(f"Stage 2: scoring {n_cache} cached competitor chunks with the stage-1 fold models"):
            for i in range(n_cache):
                Xc = pd.DataFrame(np.load(os.path.join(args.cache_dir, f"X_{i}.npy")).astype(np.float32), columns=feats)
                ids = np.load(os.path.join(args.cache_dir, f"ids_{i}.npy"))
                fr = rf[ids[1]]
                pc = np.empty(len(Xc))
                for f in np.unique(fr):
                    m = fr == f
                    # a model that never saw this record's sampled entity; all folds if it spans several
                    pc[m] = (folds_ens[f] if f >= 0 else ens_all).predict(Xc[m])
                c_s1.append(ids[0].astype(np.int32)); c_rec.append(ids[1].astype(np.int32))
                c_p.append(pc.astype(np.float32))
                for k in RAW:
                    c_raw[k].append(Xc[k].to_numpy(np.float16))
                del Xc
        shutil.rmtree(args.cache_dir, ignore_errors=True)
        all_s1 = np.concatenate([ps1.astype(np.int32)] + c_s1).astype(np.int64)
        all_rec = np.concatenate([prec.astype(np.int32)] + c_rec).astype(np.int64)
        all_p = np.concatenate([oof.astype(np.float32)] + c_p)
        raw = {k: np.concatenate([raw_E[k]] + c_raw[k]) for k in RAW}
        del raw_E
        LOG.info("Stage 2 universe: %d pairs (%d training pairs + %d competitors)", len(all_p), len(oof), len(all_p) - len(oof))
        rs = GroupStats(all_rec, all_p, len(split.rec))
        ss = GroupStats(all_s1, all_p, len(split.s1))
        n_tr = len(oof)
        ce_pack, ce_blobs = None, None
        if use_ce:
            with timer("Cross-encoder (2 models on disjoint fold halves, out-of-fold scores)"):
                ce_blobs, ce_pack = train_cross_encoder(split, ce_cfg, all_s1, all_rec, all_p, n_tr, y, pair_fold, rf,
                                                        cfg.n_jobs)
        cfg2 = copy.deepcopy(cfg.model)
        cfg2.num_boost_round, cfg2.early_stopping_rounds = 1500, 50
        cfg2.params = dict(cfg2.params, num_leaves=63, learning_rate=0.05)
        variants = [("stage2", None)] + ([("stage2+ce", ce_pack)] if ce_pack is not None else [])
        for tag, ce in variants:
            X2 = stage2_frame(all_s1, all_rec, all_p, raw, rs, ss, slice(0, n_tr), ce=ce)
            oof2 = np.zeros(len(y))
            models2 = []
            for f in range(cfg.model.n_folds):
                tr, va = pair_fold != f, pair_fold == f
                with timer(f"{tag} fold {f}"):
                    m, oof2[va], imp2 = train_fold(backend, cfg2, X2[tr], y[tr], X2[va], y[va])
                    LOG.info("  best_iter=%d  valid logloss=%.5f", m[2], log_loss(y[va], oof2[va], labels=[0, 1]))
                models2.append(m)
            LOG.info("[%s] OOF logloss %.5f (stage 1: %.5f)", tag, log_loss(y, oof2, labels=[0, 1]), oof_ll)
            if ce is not None:
                LOG.info("[%s] top features:\n%s", tag, imp2.sort_values("gain", ascending=False).head(12).to_string(index=False))
            ev = evaluate(oof2, tag)
            variant_scores[tag] = ev["nested"]
            cands[tag] = (ev, oof2, models2, list(X2.columns), ce is not None)
            if ev2 is None or ev["nested"] > ev2["nested"]:
                ev2 = ev
            del X2
        del all_s1, all_rec, all_p, raw, rs, ss, c_s1, c_rec, c_p, c_raw
    pick = max(cands, key=lambda k: cands[k][0]["nested"]) if args.final == "auto" else args.final
    if pick not in cands:
        raise SystemExit(f"--final {pick}: that variant was not trained (available: {sorted(cands)})")
    final, final_prob, s2_models, s2_features, s2_uses_ce = cands[pick]
    if not s2_uses_ce:
        ce_blobs = None
    best, sel, tuned, nested_score, nested = (final["best"], final["sel"], final["tuned"], final["nested"],
                                              final["nested_per_fold"])
    final["curve"].to_csv(os.path.join(args.model_dir, "threshold_curve.csv"), index=False)
    try:
        loss_report(split, owner, E_rows, n_true_E, ent, ps1, prec, y.astype(bool), sel.astype(bool), final_prob,
                    found[E_rows], args.model_dir)
    except Exception as ex:          # diagnostics must never cost a finished training run
        LOG.warning("loss report failed: %r", ex)
    final_tag = ("stage2+ce" if s2_uses_ce else "stage2") if s2_models else "stage1"
    LOG.info("Nested-CV macro F0.5 by variant: %s", {k: round(v, 5) for k, v in variant_scores.items()})
    LOG.info("FINAL (%s): nested-CV macro F0.5 %.5f", final_tag, nested_score)

    pd.DataFrame({"source1_entity_id": split.s1["id"].to_numpy()[ps1], "candidate_entity_id": split.rec["id"].to_numpy()[prec],
                  "label": y.astype(int), "oof_prob": oof, "final_prob": final_prob, "fold": pair_fold,
                  "selected": sel.astype(int)}) \
        .to_csv(os.path.join(args.model_dir, "oof_predictions.tsv"), sep="\t", index=False)
    report = {"blocking": blk, "oof_logloss": oof_ll, "oof_auc": oof_auc, "decision": best,
              "oof_tuned": tuned, "nested_cv_macro_f05": nested_score, "nested_per_fold": nested,
              "best_iterations": best_iters, "backend": backend, "n_features": len(feats_all),
              "train_pairs": int(len(y)), "train_entities": int(in_E.sum()),
              "stage1_nested_cv_macro_f05": ev1["nested"],
              "stage2_nested_cv_macro_f05": ev2["nested"] if ev2 else None, "uses_stage2": s2_models is not None,
              "s1_drop": s1_drop, "lexicon_size": len(lexicon), "variants": variant_scores, "final": final_tag}
    with open(os.path.join(args.model_dir, "cv_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    bundle = {"models": models, "features": feats_all, "decision": best, "config": cfg,
              "stage2_models": s2_models, "stage2_features": s2_features, "lexicon": lexicon,
              "ce_models": ce_blobs, "ce_config": config_dict(ce_cfg) if ce_blobs else None}
    with open(os.path.join(args.model_dir, "model_bundle.pkl"), "wb") as fh:
        pickle.dump(bundle, fh)
    LOG.info("Saved model bundle to %s", os.path.join(args.model_dir, "model_bundle.pkl"))
    LOG.info("Memory: %s", peak_memory_gb())


if __name__ == "__main__":
    main()
