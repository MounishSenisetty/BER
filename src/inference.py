"""End-to-end inference on a split without labels.

    python -m src.inference --data-dir <.../dataset/test> --model-dir models --out-dir output

Writes (formats exactly as required by utils/validate_submission.py):
    <out-dir>/candidate_pairs.tsv    source1_entity_id <TAB> candidate_entity_ids   (one row per S1)
    <out-dir>/matching_results.tsv   source1_entity_id <TAB> matched_entity_ids     (one row per S1)

candidate_pairs.tsv is exactly the capped candidate set the classifier scores, so every
matched id is also a candidate. If a ground truth is given (--gt), macro F0.5 is also reported.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from .features import compute_features
from .model import Ensemble
from .stage2 import RAW, predict_in_batches
from .pipeline import entity_true_counts, iter_chunks, load_split, owner_array
from .postprocess import select
from .utils import peak_memory_gb, CANDIDATE_HEADER, LOG, MATCHING_HEADER, fast_macro_f05, setup_logging, timer, write_id_lists


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/test")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--s1"), ap.add_argument("--s2"), ap.add_argument("--s3"), ap.add_argument("--gt")
    ap.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    ap.add_argument("--chunk-records", type=int, default=None)
    ap.add_argument("--dump-scores", action="store_true", help="also write pair_scores.tsv")
    args = ap.parse_args()
    setup_logging()

    with open(os.path.join(args.model_dir, "model_bundle.pkl"), "rb") as fh:
        bundle = pickle.load(fh)
    cfg = bundle["config"]
    if args.chunk_records:
        cfg.blocking.chunk_records = args.chunk_records
    rule = dict(bundle["decision"])
    if args.threshold is not None:
        rule["threshold"] = args.threshold
    LOG.info("Decision rule: mode=%s exclusive=%s threshold=%.3f", rule["mode"], rule["exclusive"], rule["threshold"])
    ens = Ensemble(bundle.get("models") or bundle["boosters"])
    feats = bundle["features"]
    ens2 = Ensemble(bundle["stage2_models"]) if bundle.get("stage2_models") else None
    LOG.info("Stage 2 re-scoring: %s", "on" if ens2 else "off")

    split = load_split(args.data_dir, args.s1, args.s2, args.s3, args.gt, with_truth=args.gt is not None)
    s1_parts, rec_parts, p_parts, cheap_parts = [], [], [], []
    raw_parts = {k: [] for k in RAW}
    for ch in iter_chunks(split, cfg):
        with timer(f"  features + scoring for {len(ch.cand)} pairs"):
            X = compute_features(ch.idx, ch.rc, ch.cand, ch.mats, n_jobs=cfg.n_jobs)
            p = ens.predict(X[feats])
        s1_parts.append(ch.g_s1.astype(np.int32))
        rec_parts.append(ch.g_rec.astype(np.int32))
        p_parts.append(p.astype(np.float32))
        cheap_parts.append(ch.cand["cheap"].to_numpy(np.float32))
        if ens2 is not None:
            for k in RAW:
                raw_parts[k].append(X[k].to_numpy(np.float16))
        del X

    empty_i = np.zeros(0, np.int32)
    ps1 = np.concatenate(s1_parts) if s1_parts else empty_i
    prec = np.concatenate(rec_parts) if rec_parts else empty_i
    prob = np.concatenate(p_parts).astype(np.float64) if p_parts else np.zeros(0)
    cheap = np.concatenate(cheap_parts) if cheap_parts else np.zeros(0, np.float32)
    s1_ids = split.s1["id"].tolist()
    rec_ids = split.rec["id"].to_numpy(dtype=object)
    LOG.info("Scored %d candidate pairs for %d records", len(prob), len(split.rec))
    if ens2 is not None:
        raw = {k: np.concatenate(v) for k, v in raw_parts.items()}
        del raw_parts
        with timer("Stage 2 re-scoring of all pairs"):
            prob = predict_in_batches(ens2, ps1.astype(np.int64), prec.astype(np.int64), prob.astype(np.float32),
                                      raw, len(split.s1), len(split.rec))
        del raw

    with timer("Writing outputs"):
        write_id_lists(os.path.join(args.out_dir, "candidate_pairs.tsv"), CANDIDATE_HEADER, s1_ids,
                       ps1, rec_ids[prec], order=cheap)
        sel = select(ps1.astype(np.int64), prec.astype(np.int64), prob, rule["mode"], rule["threshold"],
                     bool(rule["exclusive"]))
        n_nonempty = write_id_lists(os.path.join(args.out_dir, "matching_results.tsv"), MATCHING_HEADER, s1_ids,
                                    ps1[sel], rec_ids[prec[sel]], order=prob[sel])
    LOG.info("Predicted %d matches for %d / %d Source 1 entities", int(sel.sum()), n_nonempty, len(s1_ids))
    LOG.info("Memory: %s", peak_memory_gb())
    # per-country summary (the test set contains a country that is absent from training)
    ctry = split.s1["country"].to_numpy(dtype=object)
    n_pred = np.bincount(ps1[sel], minlength=len(s1_ids))
    n_cand = np.bincount(ps1, minlength=len(s1_ids))
    top = np.zeros(len(s1_ids))
    np.maximum.at(top, ps1, prob)
    summ = pd.DataFrame({"country": ctry, "n_pred": n_pred, "n_cand": n_cand, "top_p": top}).groupby("country").agg(
        entities=("n_pred", "size"), mean_matches=("n_pred", "mean"), empty_rate=("n_pred", lambda v: (v == 0).mean()),
        mean_candidates=("n_cand", "mean"), mean_top_p=("top_p", "mean"))
    LOG.info("Per-country predictions:\n%s", summ.round(4).to_string())

    if args.dump_scores:
        pd.DataFrame({"source1_entity_id": split.s1["id"].to_numpy()[ps1], "candidate_entity_id": rec_ids[prec],
                      "prob": prob, "selected": sel.astype(int)}) \
            .to_csv(os.path.join(args.out_dir, "pair_scores.tsv"), sep="\t", index=False)

    if split.truth is not None:
        owner = owner_array(split)
        n_true = entity_true_counts(split)
        y = owner[prec] == ps1
        found = np.bincount(ps1[y], minlength=len(n_true))
        LOG.info("Hold-out: pair recall of candidates %.4f | macro F0.5 %.5f",
                 found.sum() / max(n_true.sum(), 1), fast_macro_f05(ps1.astype(np.int64), sel, y, n_true))


if __name__ == "__main__":
    main()
