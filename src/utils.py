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


def peak_memory_gb() -> str:
    """Peak resident memory of this process and of its largest child (Linux: ru_maxrss in KB)."""
    try:
        import resource
        me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
        ch = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1048576
        return f"peak RSS main {me:.1f} GB, largest worker {ch:.1f} GB"
    except Exception:
        return "peak RSS unavailable"


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


def read_table(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    sep = "," if path.lower().endswith(".csv") else "\t"
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, quoting=3 if sep == "\t" else 0,
                       encoding_errors="replace", nrows=nrows)


def _pick(cols, pred, exclude=()):
    return next((c for c in cols if c not in exclude and pred(c.lower())), None)


def load_source(path: str, source: int) -> pd.DataFrame:
    """Load a source file into a canonical frame: id, name, address, country, source.

    Official schema: entity_id, business_name, business_address, country. Other layouts are
    handled heuristically (id-ish / name-ish / address-ish / country-ish headers; any remaining
    text columns are appended to the address)."""
    df = read_table(path)
    cols = list(df.columns)
    id_col = _pick(cols, lambda l: l in ("entity_id", "id")) or _pick(cols, lambda l: l.endswith("id")) or cols[0]
    name_col = _pick(cols, lambda l: "name" in l, (id_col,)) or next(c for c in cols if c != id_col)
    ctry_col = _pick(cols, lambda l: "country" in l, (id_col, name_col))
    rest = [c for c in cols if c not in (id_col, name_col, ctry_col)]
    addr_cols = [c for c in rest if "addr" in c.lower()] + [c for c in rest if "addr" not in c.lower()]

    if not addr_cols:
        addr = pd.Series([""] * len(df), index=df.index)
    elif len(addr_cols) == 1:
        addr = df[addr_cols[0]]
    else:  # vectorised join of the non-empty parts
        addr = df[addr_cols[0]].str.strip()
        for c in addr_cols[1:]:
            part = df[c].str.strip()
            addr = addr.where(part == "", addr + ", " + part).str.lstrip(", ")
    out = pd.DataFrame({
        "id": df[id_col].str.strip().to_numpy(),
        "name": df[name_col].to_numpy(),
        "address": addr.to_numpy(),
        "country": df[ctry_col].str.strip().to_numpy() if ctry_col else "",
    })
    out["source"] = np.int8(source)
    LOG.info("Loaded %s: %d rows | id=%r name=%r address=%r country=%r", os.path.basename(path), len(out),
             id_col, name_col, addr_cols, ctry_col)
    dup = int(out["id"].duplicated().sum())
    if dup:
        LOG.warning("%s has %d duplicated ids", path, dup)
    return out


def load_ground_truth(path: str, s1_ids: Iterable[str]) -> Tuple[Dict[str, Set[str]], Tuple[str, str]]:
    """Returns ({source1_id: set(matching ids)}, (s1_col_name, match_col_name)).

    Every Source 1 id is present in the result (missing rows -> singleton)."""
    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, header=None, quoting=3)
    s1_ids = list(s1_ids)
    s1_set = set(s1_ids)
    header = ("source1_entity_id", "matched_entity_ids")
    if len(raw) and raw.iloc[0, 0].strip() not in s1_set:
        header = (raw.iloc[0, 0].strip(), raw.iloc[0, 1].strip() if raw.shape[1] > 1 else "matched_entity_ids")
        raw = raw.iloc[1:]
    truth: Dict[str, Set[str]] = {i: set() for i in s1_ids}
    col1 = raw.iloc[:, 1] if raw.shape[1] > 1 else pd.Series([""] * len(raw), index=raw.index)
    for sid, m in zip(raw.iloc[:, 0], col1):
        sid = sid.strip()
        if m:
            truth.setdefault(sid, set()).update(x.strip() for x in m.split(",") if x.strip())
        else:
            truth.setdefault(sid, set())
    unknown = len(set(truth) - s1_set)
    if unknown:
        LOG.warning("ground truth has %d Source 1 ids not present in Source 1", unknown)
    return truth, header


MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def write_id_lists(path: str, header: Tuple[str, str], s1_ids: Sequence[str], pair_s1: np.ndarray,
                   pair_ids: np.ndarray, order: Optional[np.ndarray] = None) -> int:
    """One row per Source 1 entity (all of `s1_ids`, in order) with the comma-joined ids of its
    pairs, highest `order` first; empty string when an entity has none. Returns #non-empty rows."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    col = np.full(len(s1_ids), "", dtype=object)
    if len(pair_s1):
        key = np.zeros(len(pair_s1)) if order is None else -np.asarray(order, dtype=np.float64)
        o = np.lexsort((key, pair_s1))
        s_sorted, ids_sorted = np.asarray(pair_s1)[o], np.asarray(pair_ids, dtype=object)[o]
        starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
        ends = np.r_[starts[1:], len(s_sorted)]
        for st, en in zip(starts, ends):
            col[s_sorted[st]] = ",".join(dict.fromkeys(ids_sorted[st:en]))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(header) + "\n")
        fh.writelines(f"{i}\t{c}\n" for i, c in zip(s1_ids, col))
    n_nonempty = int(sum(1 for c in col if c))
    LOG.info("Wrote %s (%d rows, %d non-empty, %d ids)", path, len(col), n_nonempty, len(pair_s1))
    return n_nonempty
