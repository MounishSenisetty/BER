"""Error taxonomy on OOF predictions written by `src.train`.

    python scripts/error_analysis.py --data-dir data/dataset/train --model-dir models --top 100

Joins models/oof_predictions.tsv with the raw sources, lists the worst false positives (selected,
not a match; highest p) and false negatives (true match, not selected; lowest p), including true
pairs blocking never proposed, and buckets each error with simple heuristics:
    noise      - text noise: Devanagari, typos, case / legal-form / punctuation variants
    edge       - singleton entity / chain name / near-duplicate distractor
    sparse     - one side has an empty or near-empty address (little signal left)
    boundary   - probability within 0.25 of the tuned threshold (decision boundary)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import load_split  # noqa: E402


def bucket(r, thr, chain_names):
    b = []
    if abs(r.oof_prob - thr) < 0.25:
        b.append("boundary")
    if any(ord(c) > 0x900 for c in f"{r.name_c}{r.addr_c}"):
        b.append("noise:devanagari")
    if len(str(r.addr_c).strip()) < 5 or len(str(r.addr_s1).strip()) < 5:
        b.append("sparse:no_address")
    if r.n_true == 0:
        b.append("edge:singleton")
    if str(r.name_s1).lower() in chain_names:
        b.append("edge:chain")
    if any(w in str(r.name_c).lower().split() for w in ("express", "plus", "annex", "outlet", "ii")):
        b.append("edge:near_dup_suffix")
    return "|".join(b) or "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/dataset/train")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--top", type=int, default=100)
    a = ap.parse_args()

    split = load_split(a.data_dir, with_truth=True)
    oof = pd.read_csv(os.path.join(a.model_dir, "oof_predictions.tsv"), sep="\t", dtype={"source1_entity_id": str, "candidate_entity_id": str})
    rep = json.load(open(os.path.join(a.model_dir, "cv_report.json")))
    thr = rep["decision"]["threshold"]
    s1 = split.s1.set_index("id")
    rec = split.rec.set_index("id")
    n_true = {k: len(v) for k, v in split.truth.items()}

    # blocking misses
    cand_pairs = set(zip(oof["source1_entity_id"], oof["candidate_entity_id"]))
    miss = [(s, m) for s, ms in split.truth.items() for m in ms if (s, m) not in cand_pairs]
    miss = pd.DataFrame(miss, columns=["source1_entity_id", "candidate_entity_id"])
    miss["label"], miss["oof_prob"], miss["selected"], miss["fold"] = 1, -1.0, 0, -1

    err = pd.concat([oof[(oof.selected == 1) & (oof.label == 0)], oof[(oof.selected == 0) & (oof.label == 1)], miss], ignore_index=True)
    err["kind"] = np.where(err.label == 1, np.where(err.oof_prob < 0, "FN_blocking", "FN"), "FP")
    err["name_s1"] = s1.loc[err.source1_entity_id, "name"].to_numpy()
    err["addr_s1"] = s1.loc[err.source1_entity_id, "address"].to_numpy()
    err["name_c"] = rec.loc[err.candidate_entity_id, "name"].to_numpy()
    err["addr_c"] = rec.loc[err.candidate_entity_id, "address"].to_numpy()
    err["n_true"] = err.source1_entity_id.map(n_true)
    names = split.s1["name"].str.lower().value_counts()
    chain_names = set(names[names >= 5].index)
    err["bucket"] = [bucket(r, thr, chain_names) for r in err.itertuples()]

    print(f"threshold={thr}  errors: {err.kind.value_counts().to_dict()}")
    print("\nBucket counts by kind (multi-label):")
    ex = err.assign(b=err.bucket.str.split("|")).explode("b", ignore_index=True)
    print(pd.crosstab(ex.b, ex.kind, margins=True).to_string())
    pd.set_option("display.width", 250), pd.set_option("display.max_colwidth", 60)
    cols = ["oof_prob", "n_true", "name_s1", "name_c", "addr_s1", "addr_c", "bucket"]
    fp = err[err.kind == "FP"].sort_values("oof_prob", ascending=False).head(a.top)
    fn = err[err.kind != "FP"].sort_values("oof_prob").head(a.top)
    print(f"\n=== worst {len(fp)} false positives ===\n{fp[cols].to_string(index=False)}")
    print(f"\n=== worst {len(fn)} false negatives ===\n{fn[cols].to_string(index=False)}")
    err.to_csv(os.path.join(a.model_dir, "errors.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
