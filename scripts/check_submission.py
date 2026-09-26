"""Local integrity check of the two deliverables (a stand-in for utils/validate_submission.py,
which ships with the competition resources -- run that one too before submitting).

    python scripts/check_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir <.../dataset/test>

Checks: exact headers; exactly one row per test Source 1 entity (no missing / extra / duplicated
ids), per country; every listed id exists in Source 2 or Source 3 and appears once per row; every
matched id is also a candidate of the same entity. Prints PASS and exits 0, else FAIL and exits 1.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import CANDIDATE_HEADER, MATCHING_HEADER, find_file, load_source  # noqa: E402


def read_lists(path: str, header, errors: list) -> dict:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    if tuple(df.columns) != tuple(header):
        errors.append(f"{os.path.basename(path)}: header {tuple(df.columns)} != {header}")
        return {}
    dup = df.iloc[:, 0][df.iloc[:, 0].duplicated()]
    if len(dup):
        errors.append(f"{os.path.basename(path)}: {len(dup)} duplicated Source 1 rows, e.g. {dup.iloc[0]}")
    out = {}
    for sid, ids in zip(df.iloc[:, 0], df.iloc[:, 1]):
        lst = [x for x in ids.split(",") if x] if ids else []
        if len(lst) != len(set(lst)):
            errors.append(f"{os.path.basename(path)}: duplicated ids in the row of {sid}")
        out[sid] = lst
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    a = ap.parse_args()

    s1 = load_source(find_file(a.test_dir, "s1"), 1)
    rec_ids = set(load_source(find_file(a.test_dir, "s2"), 2)["id"]) | set(load_source(find_file(a.test_dir, "s3"), 3)["id"])
    errors: list = []
    match = read_lists(a.matching, MATCHING_HEADER, errors)
    cand = read_lists(a.candidate, CANDIDATE_HEADER, errors)

    s1_ids = set(s1["id"])
    for name, d in (("matching", match), ("candidate", cand)):
        missing, extra = s1_ids - set(d), set(d) - s1_ids
        if missing:
            errors.append(f"{name}: {len(missing)} Source 1 entities missing, e.g. {next(iter(missing))}")
        if extra:
            errors.append(f"{name}: {len(extra)} unknown Source 1 ids, e.g. {next(iter(extra))}")
        unknown = {x for v in d.values() for x in v} - rec_ids
        if unknown:
            errors.append(f"{name}: {len(unknown)} ids not in Source 2/3, e.g. {next(iter(unknown))}")
    not_cand = [(s, m) for s, ms in match.items() for m in ms if m not in set(cand.get(s, ()))]
    if not_cand:
        errors.append(f"{len(not_cand)} matched ids are not candidates of their entity, e.g. {not_cand[0]}")

    per_country = s1.assign(present=s1["id"].isin(match.keys()),
                            matched=s1["id"].map(lambda i: bool(match.get(i))))
    print(per_country.groupby("country").agg(s1_rows=("id", "size"), present=("present", "sum"),
                                             with_matches=("matched", "sum")).to_string())
    print(f"matched pairs {sum(map(len, match.values()))} | candidate pairs {sum(map(len, cand.values()))}")
    if errors:
        print("FAIL\n  " + "\n  ".join(errors))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
