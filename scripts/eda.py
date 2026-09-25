"""Dataset analysis for the competition files. Run this before training.

    python scripts/eda.py --train-dir <.../dataset/train> --test-dir <.../dataset/test>

Prints the raw schema, sizes, missing values and ID formats of every source, ground-truth
statistics, and side-by-side examples of true matches. It also checks the assumptions the
pipeline relies on:
  * IDs are unique within and across sources (the output joins them with commas)
  * every S2/S3 record belongs to at most one S1 entity (the "exclusivity" rule)
  * which address atoms exist (postal code length -> US ZIP vs Indian PIN, house numbers)
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.normalize import LEGAL_SUFFIXES, clean  # noqa: E402
from src.utils import find_file, load_ground_truth, load_source, read_table  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 70)


def hr(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def id_shape(ids):
    return Counter(re.sub(r"[0-9]", "9", re.sub(r"[A-Za-z]", "a", i)) for i in ids).most_common(3)


def describe_raw(path):
    df = read_table(path)
    print(f"\n--- {os.path.basename(path)}: {len(df):,} rows x {df.shape[1]} cols")
    print("columns:", list(df.columns))
    print("empty-string rate per column:")
    print((df == "").mean().round(4).to_string())
    print(df.head(5).to_string(index=False))
    return df


def core(s):
    t = [x for x in clean(s).split() if x not in LEGAL_SUFFIXES]
    return " ".join(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--n-examples", type=int, default=15)
    a = ap.parse_args()
    rnd = random.Random(0)

    hr("1. RAW FILES")
    paths = {}
    for split, d in [("train", a.train_dir), ("test", a.test_dir)]:
        if not d:
            continue
        for k in ["s1", "s2", "s3"] + (["gt"] if split == "train" else []):
            p = find_file(d, k)
            paths[(split, k)] = p
            print(f"{split:5s} {k}: {p}")
    for key, p in paths.items():
        if p and key[1] != "gt":
            describe_raw(p)

    hr("2. CANONICAL VIEW (what the pipeline sees after column detection)")
    s1 = load_source(paths[("train", "s1")], 1)
    s2 = load_source(paths[("train", "s2")], 2)
    s3 = load_source(paths[("train", "s3")], 3)
    rec = pd.concat([s2, s3], ignore_index=True)
    for nm, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(f"{nm}: rows={len(df):,} dup_ids={df['id'].duplicated().sum()} id_shapes={id_shape(df['id'])} "
              f"empty_name={(df['name'].str.strip() == '').mean():.3f} empty_addr={(df['address'].str.strip() == '').mean():.3f} "
              f"name_len={df['name'].str.len().median():.0f} addr_len={df['address'].str.len().median():.0f}")
    print(f"ID collisions S1∩S2={len(set(s1.id) & set(s2.id))}  S1∩S3={len(set(s1.id) & set(s3.id))}  "
          f"S2∩S3={len(set(s2.id) & set(s3.id))}   (must be 0 for unambiguous output)")

    hr("3. GROUND TRUTH")
    gt_path = paths[("train", "gt")]
    with open(gt_path, encoding="utf8", errors="replace") as fh:
        print("first lines:\n" + "".join(fh.readline() for _ in range(4)))
    truth, header = load_ground_truth(gt_path, s1["id"])
    print("header detected:", header)
    sizes = pd.Series({k: len(v) for k, v in truth.items()})
    print(f"entities in GT: {len(truth):,} | S1 rows: {len(s1):,} | GT ids missing from S1: {len(set(truth) - set(s1.id))}")
    print(f"singleton rate: {(sizes == 0).mean():.3%}")
    print("matches per entity:\n" + sizes.clip(upper=10).value_counts().sort_index().to_string())
    all_m = [m for v in truth.values() for m in v]
    src_of = dict(zip(rec["id"], rec["source"]))
    print(f"true pairs: {len(all_m):,} | from S2: {sum(src_of.get(m) == 2 for m in all_m):,} | "
          f"from S3: {sum(src_of.get(m) == 3 for m in all_m):,} | unknown ids: {sum(m not in src_of for m in all_m):,}")
    owners = Counter(all_m)
    multi = sum(1 for c in owners.values() if c > 1)
    print(f"records owned by >1 entity: {multi:,}  -> exclusivity assumption {'HOLDS' if multi == 0 else 'VIOLATED'}")
    matched = set(all_m)
    print(f"S2 records matched: {s2['id'].isin(matched).mean():.3%} | S3 records matched: {s3['id'].isin(matched).mean():.3%} "
          f"(the rest are unmatched distractors)")

    hr("4. HOW NOISY ARE TRUE MATCHES?")
    s1i, reci = s1.set_index("id"), rec.set_index("id")
    pairs = [(s, m) for s, ms in truth.items() for m in ms if s in s1i.index and m in reci.index]
    if pairs:
        from rapidfuzz import fuzz
        smp = rnd.sample(pairs, min(5000, len(pairs)))
        eq = sum(core(s1i.at[s, "name"]) == core(reci.at[m, "name"]) for s, m in smp) / len(smp)
        tsr = pd.Series([fuzz.token_set_ratio(core(s1i.at[s, "name"]), core(reci.at[m, "name"])) for s, m in smp])
        asr = pd.Series([fuzz.token_set_ratio(clean(s1i.at[s, "address"]), clean(reci.at[m, "address"])) for s, m in smp])
        print(f"normalised core-name exact match among true pairs: {eq:.3%}")
        print("name token_set_ratio quantiles:", tsr.quantile([.05, .1, .25, .5]).round(1).to_dict())
        print("addr token_set_ratio quantiles:", asr.quantile([.05, .1, .25, .5]).round(1).to_dict())
        print("\nExamples of true matches (S1 | record):")
        for s, m in rnd.sample(pairs, min(a.n_examples, len(pairs))):
            print(f"  S1 : {s1i.at[s, 'name']!r:45} | {s1i.at[s, 'address']!r}")
            print(f"  S{reci.at[m, 'source']} : {reci.at[m, 'name']!r:45} | {reci.at[m, 'address']!r}\n")

    hr("5. ADDRESS FORMAT")
    nums = Counter(len(t) for x in s1["address"] for t in re.findall(r"\b\d+\b", x))
    print("length of numeric tokens in S1 addresses (5 -> US ZIP, 6 -> Indian PIN):", dict(sorted(nums.items())))
    toks = Counter(t for x in s1["address"].head(50000) for t in clean(x).split() if not t.isdigit())
    print("most common address tokens:", toks.most_common(40))
    ntoks = Counter(t for x in s1["name"].head(50000) for t in clean(x).split())
    print("most common name tokens:", ntoks.most_common(40))
    dup_names = s1["name"].map(core).value_counts()
    print(f"S1 core names shared by >1 entity (chains): {(dup_names > 1).sum():,}; top: {dup_names.head(10).to_dict()}")

    if a.test_dir:
        hr("6. TEST SPLIT")
        t1 = load_source(paths[("test", "s1")], 1)
        t2 = load_source(paths[("test", "s2")], 2)
        t3 = load_source(paths[("test", "s3")], 3)
        print(f"test S1={len(t1):,} S2={len(t2):,} S3={len(t3):,} | train S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}")
        print(f"test ids overlapping train ids: {len((set(t1.id) | set(t2.id) | set(t3.id)) & (set(s1.id) | set(rec.id)))}")
        print(f"record/entity ratio train={len(rec) / len(s1):.2f} test={(len(t2) + len(t3)) / max(len(t1), 1):.2f}")


if __name__ == "__main__":
    main()
