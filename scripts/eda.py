"""Dataset analysis for the competition files. Run this before training (takes a few minutes).

    python scripts/eda.py --train-dir <.../dataset/train> --test-dir <.../dataset/test>

Prints sizes / countries / scripts / missing values per source, ground-truth statistics and
checks of the assumptions the pipeline relies on:
  * IDs are unique across sources and carry the S1-/S2-/S3- prefix
  * every S2/S3 record belongs to at most one S1 entity ("exclusivity")
  * matches never cross countries (blocking is partitioned by country)
plus noise levels of true matches and side-by-side examples (raw and normalised).
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.normalize import _prep_one  # noqa: E402
from src.utils import find_file, load_ground_truth, load_source  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 80)
_RE_INDIC = re.compile("[ऀ-෿]")
_RE_NONASCII = re.compile(r"[^\x00-\x7f]")


def hr(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100, flush=True)


def profile(nm, df):
    print(f"\n{nm}: {len(df):,} rows | dup ids {df['id'].duplicated().sum()} | "
          f"prefixes {Counter(i[:3] for i in df['id'].head(100000)).most_common(3)}")
    g = df.groupby("country")
    t = pd.DataFrame({
        "rows": g.size(),
        "empty_addr": g["address"].apply(lambda s: (s.str.strip() == "").mean()).round(4),
        "null_in_addr": g["address"].apply(lambda s: s.str.contains(r"\bnull\b", case=False).mean()).round(4),
        "indic_name": g["name"].apply(lambda s: s.str.contains(_RE_INDIC).mean()).round(4),
        "indic_addr": g["address"].apply(lambda s: s.str.contains(_RE_INDIC).mean()).round(4),
        "nonascii_name": g["name"].apply(lambda s: s.str.contains(_RE_NONASCII).mean()).round(4),
        "name_len": g["name"].apply(lambda s: s.str.len().median()),
        "addr_len": g["address"].apply(lambda s: s.str.len().median()),
    })
    print(t.to_string())


def section(fn):
    """Run one report section; print the error and continue instead of aborting the report."""
    def wrapped(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"!! section failed: {e!r}\n{traceback.format_exc()}", flush=True)
    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--n-examples", type=int, default=12)
    a = ap.parse_args()
    rnd = random.Random(0)

    hr("1. FILES AND PER-SOURCE PROFILE (train)")
    s1 = load_source(find_file(a.train_dir, "s1"), 1)
    s2 = load_source(find_file(a.train_dir, "s2"), 2)
    s3 = load_source(find_file(a.train_dir, "s3"), 3)
    for nm, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        profile(nm, df)
    rec = pd.concat([s2, s3], ignore_index=True)
    print(f"\nID collisions S1∩S2={len(set(s1.id) & set(s2.id))} S1∩S3={len(set(s1.id) & set(s3.id))} "
          f"S2∩S3={len(set(s2.id) & set(s3.id))}")

    hr("2. GROUND TRUTH")
    gt_path = find_file(a.train_dir, "gt")
    with open(gt_path, encoding="utf8", errors="replace") as fh:
        print("first lines:\n" + "".join(fh.readline() for _ in range(3)))
    truth, header = load_ground_truth(gt_path, s1["id"])
    print("header:", header)
    n_true = pd.Series({k: len(v) for k, v in truth.items()})
    ctry1 = dict(zip(s1["id"], s1["country"]))
    print(f"entities {len(truth):,} | singleton rate {(n_true == 0).mean():.2%} | mean matches {n_true.mean():.2f} "
          f"| max {n_true.max()}")
    print("matches per entity (clipped at 10):\n" + n_true.clip(upper=10).value_counts().sort_index().to_string())
    by_c = pd.DataFrame({"n": n_true, "c": [ctry1.get(i, "?") for i in n_true.index]}).groupby("c")["n"]
    print("per country: singleton rate / mean matches\n" +
          pd.DataFrame({"singleton": by_c.apply(lambda s: (s == 0).mean()).round(4), "mean": by_c.mean().round(3)}).to_string())
    pairs = [(s, m) for s, ms in truth.items() for m in ms]
    src_of = dict(zip(rec["id"], rec["source"]))
    ctry2 = dict(zip(rec["id"], rec["country"]))
    print(f"true pairs {len(pairs):,} | S2 {sum(src_of.get(m) == 2 for _, m in pairs):,} | "
          f"S3 {sum(src_of.get(m) == 3 for _, m in pairs):,} | unknown ids {sum(m not in src_of for _, m in pairs):,}")
    owners = Counter(m for _, m in pairs)
    multi = sum(1 for c in owners.values() if c > 1)
    print(f"records owned by >1 entity: {multi:,} -> exclusivity {'HOLDS' if multi == 0 else 'VIOLATED'}")
    cross = sum(1 for s, m in pairs if m in ctry2 and ctry1.get(s) != ctry2[m])
    print(f"cross-country true pairs: {cross:,} ({cross / max(len(pairs), 1):.3%}) -> country partitioning "
          f"{'SAFE' if cross == 0 else 'loses this share of recall'}")
    matched = set(owners)
    print(f"records matched: S2 {s2['id'].isin(matched).mean():.2%} | S3 {s3['id'].isin(matched).mean():.2%} "
          f"(rest are unmatched distractors)")

    s1i, reci = s1.set_index("id"), rec.set_index("id")

    @section
    def sec_noise():
        hr("3. NOISE IN TRUE MATCHES (sample of 20k pairs, after normalisation)")
        from rapidfuzz import fuzz
        smp = rnd.sample(pairs, min(20000, len(pairs)))
        rows = []
        for s, m in smp:
            if s not in s1i.index or m not in reci.index:
                continue
            A, B = s1i.loc[s], reci.loc[m]
            pa, pb = _prep_one(A["name"], A["address"], A["country"]), _prep_one(B["name"], B["address"], B["country"])
            rows.append({"s": s, "m": m, "country": A["country"], "src": B["source"], "core_eq": float(pa[1] == pb[1]),
                         "name_tset": fuzz.token_set_ratio(pa[1], pb[1]), "addr_tset": fuzz.token_set_ratio(pa[3], pb[3]),
                         "addr_missing": float(pb[3] == ""),
                         "house_eq": float(pa[16] == pb[16]) if pa[16] and pb[16] else np.nan,
                         "postal_eq": float(pa[14] == pb[14]) if pa[14] and pb[14] else np.nan,
                         "indic": float(pb[18])})
        d = pd.DataFrame(rows)
        g = d.groupby(["country", "src"])
        print(pd.DataFrame({"n": g.size(), "core_eq": g["core_eq"].mean().round(3),
                            "name_tset_p10": g["name_tset"].quantile(.1), "name_tset_med": g["name_tset"].median(),
                            "addr_tset_p10": g["addr_tset"].quantile(.1), "addr_missing": g["addr_missing"].mean().round(3),
                            "house_eq": g["house_eq"].mean().round(3), "postal_eq": g["postal_eq"].mean().round(3),
                            "indic_name": g["indic"].mean().round(3)}).to_string())
        print("\nHardest true matches (lowest normalised name similarity):")
        for _, r in d.nsmallest(15, "name_tset").iterrows():
            A, B = s1i.loc[r["s"]], reci.loc[r["m"]]
            print(f"  [{r['name_tset']:.0f}] S1 {A['name']!r} | {A['address']!r}\n"
                  f"        S{B['source']} {B['name']!r} | {B['address']!r}")

    sec_noise()

    @section
    def sec_examples():
        hr("4. EXAMPLES OF TRUE MATCHES (raw | normalised core name)")
        for c in sorted(s1["country"].unique()):
            cp = [(s, m) for s, m in pairs if ctry1.get(s) == c]
            print(f"\n--- {c} ---")
            for s, m in rnd.sample(cp, min(a.n_examples, len(cp))):
                A, B = s1i.loc[s], reci.loc[m]
                print(f"  S1 : {A['name']!r:50} | {A['address']!r}\n       -> {_prep_one(A['name'], A['address'], c)[1]!r}")
                print(f"  S{B['source']} : {B['name']!r:50} | {B['address']!r}\n       -> {_prep_one(B['name'], B['address'], c)[1]!r}\n")

    sec_examples()

    @section
    def sec_chains():
        hr("5. CHAINS / DUPLICATE NAMES IN SOURCE 1")
        core = s1.sample(min(300000, len(s1)), random_state=0)
        core = core.assign(core=[_prep_one(n, "", c)[1] for n, c in zip(core["name"], core["country"])])
        vc = core.groupby("country")["core"].value_counts()
        for c in core["country"].unique():
            top = vc[c].head(12)
            print(f"{c}: share of names shared by >1 entity {(vc[c] > 1).sum() / max(len(vc[c]), 1):.2%} | top {top.to_dict()}")

        if a.test_dir:
            hr("6. TEST SPLIT")
            t1 = load_source(find_file(a.test_dir, "s1"), 1)
            t2 = load_source(find_file(a.test_dir, "s2"), 2)
            t3 = load_source(find_file(a.test_dir, "s3"), 3)
            for nm, df in [("test S1", t1), ("test S2", t2), ("test S3", t3)]:
                profile(nm, df)
            print(f"\nrecords per S1 entity: train {len(rec) / len(s1):.2f} | test {(len(t2) + len(t3)) / max(len(t1), 1):.2f}")

    sec_chains()


if __name__ == "__main__":
    main()
