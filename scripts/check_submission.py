"""Format / sanity checks for the two deliverables (complements the official validator).

    python scripts/check_submission.py --data-dir <.../dataset/test> --out-dir output

Checks
  matching_results.tsv : tab-separated, 2 columns, header, every S1 id exactly once, no unknown
                         S1 ids, every predicted id exists in S2/S3, no duplicates in a row,
                         no record predicted for two entities
  candidate_pairs.tsv  : ids exist, and every predicted match is also a candidate pair
Exit code 1 if any hard check fails.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import find_file, load_source  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="output")
    a = ap.parse_args()
    errors, warns = [], []

    s1 = load_source(find_file(a.data_dir, "s1"), 1)
    rec = pd.concat([load_source(find_file(a.data_dir, "s2"), 2), load_source(find_file(a.data_dir, "s3"), 3)])
    s1_ids, rec_ids = set(s1["id"]), set(rec["id"])

    res_path = os.path.join(a.out_dir, "matching_results.tsv")
    with open(res_path, encoding="utf8") as fh:
        lines = fh.read().splitlines()
    print(f"{res_path}: {len(lines) - 1:,} data rows, header={lines[0].split(chr(9))}")
    bad_cols = [i for i, l in enumerate(lines) if len(l.split("\t")) != 2]
    if bad_cols:
        errors.append(f"{len(bad_cols)} lines do not have exactly 2 tab-separated fields (first: line {bad_cols[0] + 1})")
    rows = [l.split("\t") for l in lines[1:] if len(l.split("\t")) == 2]
    ids = [r[0] for r in rows]
    cnt = Counter(ids)
    if missing := s1_ids - set(ids):
        errors.append(f"{len(missing)} Source 1 ids missing from the results")
    if extra := set(ids) - s1_ids:
        errors.append(f"{len(extra)} ids in the results are not Source 1 ids")
    if dups := [i for i, c in cnt.items() if c > 1]:
        errors.append(f"{len(dups)} Source 1 ids appear more than once")
    owner = Counter()
    n_pred, n_nonempty = 0, 0
    for sid, m in rows:
        ms = [x for x in m.split(",") if x]
        if m != m.strip() or any(x != x.strip() for x in ms):
            errors.append(f"whitespace inside match list for {sid}")
            break
        if len(ms) != len(set(ms)):
            errors.append(f"duplicate ids in the match list of {sid}")
        if unknown := [x for x in ms if x not in rec_ids]:
            errors.append(f"{sid}: {len(unknown)} predicted ids not in Source 2/3 (e.g. {unknown[0]})")
        owner.update(set(ms))
        n_pred += len(ms)
        n_nonempty += bool(ms)
    if shared := sum(1 for c in owner.values() if c > 1):
        warns.append(f"{shared} records are predicted for more than one S1 entity")
    print(f"predicted matches: {n_pred:,} | entities with >=1 match: {n_nonempty:,} "
          f"({n_nonempty / max(len(rows), 1):.1%}) | predicted singletons: {len(rows) - n_nonempty:,}")

    cp_path = os.path.join(a.out_dir, "candidate_pairs.tsv")
    if os.path.exists(cp_path):
        cp = pd.read_csv(cp_path, sep="\t", dtype=str, keep_default_na=False)
        print(f"{cp_path}: {len(cp):,} pairs, columns={list(cp.columns)}, "
              f"{len(cp) / max(len(rec), 1):.1f} per record")
        if (~cp.iloc[:, 0].isin(s1_ids)).any() or (~cp.iloc[:, 1].isin(rec_ids)).any():
            errors.append("candidate_pairs.tsv contains unknown ids")
        cand = set(zip(cp.iloc[:, 0], cp.iloc[:, 1]))
        outside = sum(1 for sid, m in rows for x in m.split(",") if x and (sid, x) not in cand)
        if outside:
            errors.append(f"{outside} predicted matches are not in candidate_pairs.tsv")
    else:
        errors.append(f"{cp_path} not found")

    for w in warns:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    print("RESULT:", "FAILED" if errors else "OK")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
