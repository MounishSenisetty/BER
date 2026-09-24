"""I/O helpers and the exact competition metric (macro F0.5 over Source 1 entities)."""
from __future__ import annotations

import glob
import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

BETA2 = 0.25  # beta = 0.5  ->  beta^2 = 0.25

LOG = logging.getLogger("ber")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")


@contextmanager
def timer(msg: str):
    t0 = time.time()
    LOG.info("%s ...", msg)
    yield
    LOG.info("%s done in %.1fs", msg, time.time() - t0)


# ----------------------------------------------------------------------------------------
# Metric
# ----------------------------------------------------------------------------------------
def entity_f05(true_set: Set[str], pred_set: Set[str]) -> float:
    """F0.5 for one Source 1 entity, with the competition's singleton rule.

    * no true match & empty prediction      -> 1.0
    * no true match & any prediction        -> 0.0
    * true matches  & empty prediction      -> 0.0
    * otherwise F0.5 = 1.25 P R / (0.25 P + R) = 1.25 TP / (0.25 |T| + |P|)
    """
    if not true_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    p = tp / len(pred_set)
    r = tp / len(true_set)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(truth: Mapping[str, Set[str]], pred: Mapping[str, Iterable[str]],
              entities: Optional[Sequence[str]] = None) -> float:
    """Mean per-entity F0.5. Entities default to every key of `truth` (i.e. every Source 1
    row of the ground-truth file); entities missing from `pred` count as an empty prediction."""
    ents = list(truth.keys()) if entities is None else list(entities)
    if not ents:
        return 0.0
    return float(np.mean([entity_f05(set(truth.get(e, ())), set(pred.get(e, ()))) for e in ents]))


def macro_f05_breakdown(truth: Mapping[str, Set[str]], pred: Mapping[str, Iterable[str]]) -> Dict[str, float]:
    single, multi = [], []
    for e, t in truth.items():
        s = entity_f05(set(t), set(pred.get(e, ())))
        (multi if t else single).append(s)
    allv = single + multi
    return {
        "macro_f05": float(np.mean(allv)) if allv else 0.0,
        "singleton_f05": float(np.mean(single)) if single else float("nan"),
        "matched_f05": float(np.mean(multi)) if multi else float("nan"),
        "n_singletons": len(single),
        "n_matched": len(multi),
    }


def fast_macro_f05(ent_idx: np.ndarray, selected: np.ndarray, label: np.ndarray,
                   n_true: np.ndarray) -> float:
    """Vectorised macro F0.5 used inside threshold searches.

    ent_idx  : entity index (0..E-1) of each candidate pair
    selected : bool, pair predicted as a match
    label    : bool, pair is a true match
    n_true   : int array (E,), size of each entity's full true set (incl. blocking misses)
    """
    E = len(n_true)
    tp = np.bincount(ent_idx, weights=(selected & label).astype(np.float64), minlength=E)
    npred = np.bincount(ent_idx, weights=selected.astype(np.float64), minlength=E)
    denom = BETA2 * n_true + npred
    f = np.where(denom > 0, (1 + BETA2) * tp / np.maximum(denom, 1e-12), 1.0)  # 0/0 -> singleton hit
    return float(f.mean())


# ----------------------------------------------------------------------------------------
# File discovery / loading
# ----------------------------------------------------------------------------------------
_SOURCE_PATTERNS = {
    "s1": ["*source*1*", "*source_1*", "*src1*", "*s1*"],
    "s2": ["*source*2*", "*source_2*", "*src2*", "*s2*"],
    "s3": ["*source*3*", "*source_3*", "*src3*", "*s3*"],
    "gt": ["*ground*truth*", "*truth*", "*labels*"],
}


def find_file(data_dir: str, key: str) -> Optional[str]:
    for pat in _SOURCE_PATTERNS[key]:
        hits = sorted(h for h in glob.glob(os.path.join(data_dir, pat))
                      if h.lower().endswith((".tsv", ".csv", ".txt")))
        if key != "gt":
            hits = [h for h in hits if "truth" not in os.path.basename(h).lower()]
        if hits:
            return hits[0]
    return None


def read_table(path: str) -> pd.DataFrame:
    sep = "," if path.lower().endswith(".csv") else "\t"
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, quoting=3 if sep == "\t" else 0,
                       encoding_errors="replace")


_ADDR_HINTS = ("addr", "street", "city", "state", "zip", "postal", "pin", "country",
               "locality", "area", "line", "district", "region", "province", "suburb", "town")


def load_source(path: str, source: int) -> pd.DataFrame:
    """Load a source file into a canonical frame: id, name, address, source.

    Column detection is heuristic (id-ish / name-ish / address-ish headers); every
    remaining text column is folded into the address so no signal is discarded."""
    df = read_table(path)
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    id_col = next((c for c, l in zip(cols, low) if l in ("id", "entity_id", "record_id")), None) \
        or next((c for c, l in zip(cols, low) if l.endswith("id")), None) \
        or next((c for c, l in zip(cols, low) if "id" in l), cols[0])
    name_col = next((c for c, l in zip(cols, low) if "name" in l and c != id_col), None) \
        or next(c for c in cols if c != id_col)
    rest = [c for c in cols if c not in (id_col, name_col)]
    addr_cols = [c for c in rest if any(h in c.lower() for h in _ADDR_HINTS)] or rest

    addr = df[addr_cols].astype(str).agg(lambda r: ", ".join(x.strip() for x in r if x and x.strip()), axis=1) \
        if addr_cols else pd.Series([""] * len(df), index=df.index)
    out = pd.DataFrame({
        "id": df[id_col].astype(str).str.strip().to_numpy(),
        "name": df[name_col].astype(str).to_numpy(),
        "address": addr.astype(str).to_numpy(),
    })
    out["source"] = source
    LOG.info("Loaded %s: %d rows | id=%r name=%r address=%r", os.path.basename(path), len(out),
             id_col, name_col, addr_cols)
    dup = out["id"].duplicated().sum()
    if dup:
        LOG.warning("%s has %d duplicated ids", path, dup)
    return out


def load_ground_truth(path: str, s1_ids: Iterable[str]) -> Tuple[Dict[str, Set[str]], Tuple[str, str]]:
    """Returns ({source1_id: set(matching ids)}, (s1_col_name, match_col_name)).

    Every Source 1 id is present in the result (missing rows -> singleton)."""
    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, header=None, quoting=3)
    s1_ids = list(s1_ids)
    s1_set = set(s1_ids)
    header = ("source1_id", "matches")
    if len(raw) and raw.iloc[0, 0].strip() not in s1_set:
        header = (raw.iloc[0, 0].strip(), raw.iloc[0, 1].strip() if raw.shape[1] > 1 else "matches")
        raw = raw.iloc[1:]
    truth: Dict[str, Set[str]] = {i: set() for i in s1_ids}
    col1 = raw.iloc[:, 1] if raw.shape[1] > 1 else pd.Series([""] * len(raw), index=raw.index)
    for sid, m in zip(raw.iloc[:, 0], col1):
        sid = sid.strip()
        ms = {x.strip() for x in str(m).split(",") if x.strip()}
        truth.setdefault(sid, set()).update(ms)
    unknown = len(set(truth) - s1_set)
    if unknown:
        LOG.warning("ground truth has %d Source 1 ids not present in Source 1", unknown)
    return truth, header


def write_matching_results(path: str, s1_ids: Sequence[str], pred: Mapping[str, Sequence[str]],
                           header: Tuple[str, str] = ("source1_id", "matches")) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [(i, ",".join(pred.get(i, ()))) for i in s1_ids]
    pd.DataFrame(rows, columns=list(header)).to_csv(path, sep="\t", index=False)
    LOG.info("Wrote %s (%d rows, %d non-empty)", path, len(rows), sum(1 for _, m in rows if m))


def write_candidate_pairs(path: str, pairs: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pairs[["source1_id", "candidate_id", "candidate_source"]].to_csv(path, sep="\t", index=False)
    LOG.info("Wrote %s (%d pairs)", path, len(pairs))
