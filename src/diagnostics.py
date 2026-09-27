"""Where the macro-F0.5 points are lost (printed at the end of training, saved as loss_report.tsv).

For every sampled entity the OOF prediction is compared with the truth, and the loss is split into
what fixing each error type alone would gain:

  FP        predicted pairs that are wrong: a distractor record (its business is not in Source 1)
            or a record that belongs to another Source 1 entity
  FN        true pairs that were scored but not selected
  MISS      true pairs that blocking never proposed (not in candidate_pairs)

Each error pair also gets descriptive tags (record without address, Source 1 name shared by several
entities, native-script record name, name similarity band), so the report says which part is
irreducible (an address-less record whose name belongs to five Source 1 entities) and which part a
better model could win back.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from .utils import LOG

_RE_INDIC = re.compile("[\u0900-\u0DFF]")


def _simple(s: pd.Series) -> pd.Series:
    return s.fillna("").str.lower().str.replace(r"[^0-9a-z\u0900-\u0dff]+", "", regex=True)


def loss_report(split, owner: np.ndarray, E_rows: np.ndarray, n_true_E: np.ndarray, ent: np.ndarray,
                ps1: np.ndarray, prec: np.ndarray, y: np.ndarray, sel: np.ndarray, prob: np.ndarray,
                found_E: np.ndarray, model_dir: str, n_show: int = 15) -> None:
    E = len(E_rows)
    tp = np.bincount(ent, weights=(sel & y), minlength=E)
    npred = np.bincount(ent, weights=sel, minlength=E)
    nfp = npred - tp
    nfn = np.bincount(ent, weights=(~sel & y), minlength=E)
    miss = np.maximum(n_true_E - found_E, 0)
    T = n_true_E

    def f(tp_, p_):
        d = 0.25 * T + p_
        return np.where(d > 0, 1.25 * tp_ / np.maximum(d, 1e-12), 1.0)

    base = f(tp, npred)
    gain = {"FP": f(tp, tp) - base, "FN": f(tp + nfn, npred + nfn) - base, "MISS": f(tp + miss, npred + miss) - base}
    LOG.info("Loss report: macro F0.5 %.5f; points (x100) recovered by fixing only ... FP %.3f | scored FN %.3f | "
             "blocking misses %.3f", base.mean(), 100 * gain["FP"].mean(), 100 * gain["FN"].mean(),
             100 * gain["MISS"].mean())

    # --- per-pair attribution of each entity's gain
    fp_i = np.flatnonzero(sel & ~y)
    fn_i = np.flatnonzero(~sel & y)
    rows = []
    s1_rows_E = E_rows
    for kind, idx, cnt in (("FP", fp_i, nfp), ("FN", fn_i, nfn)):
        e = ent[idx]
        rows.append(pd.DataFrame({"kind": kind, "s1": ps1[idx], "rec": prec[idx], "p": prob[idx],
                                  "pts": gain[kind][e] / np.maximum(cnt[e], 1)}))
    # blocking misses: every true record of a sampled entity that is not among its candidates
    cand_true = set(zip(ps1[y].tolist(), prec[y].tolist()))
    in_E = np.zeros(len(split.s1), dtype=bool)
    in_E[s1_rows_E] = True
    e_of = np.full(len(split.s1), -1, dtype=np.int64)
    e_of[s1_rows_E] = np.arange(E)
    rr = np.flatnonzero((owner >= 0) & in_E[np.maximum(owner, 0)])
    rr = np.array([r for r in rr if (owner[r], r) not in cand_true], dtype=np.int64)
    if len(rr):
        e = e_of[owner[rr]]
        rows.append(pd.DataFrame({"kind": "MISS", "s1": owner[rr], "rec": rr, "p": np.nan,
                                  "pts": gain["MISS"][e] / np.maximum(miss[e], 1)}))
    err = pd.concat(rows, ignore_index=True)
    if err.empty:
        return
    S1, R = split.s1, split.rec
    s1_name = S1["name"].to_numpy(dtype=object)
    s1_simple = _simple(S1["name"])
    dup = s1_simple.map(s1_simple.value_counts()).to_numpy()
    err["s1_name"] = s1_name[err.s1]
    err["s1_address"] = S1["address"].to_numpy(dtype=object)[err.s1]
    err["rec_name"] = R["name"].to_numpy(dtype=object)[err.rec]
    err["rec_address"] = R["address"].to_numpy(dtype=object)[err.rec]
    from rapidfuzz import fuzz, process
    a = _simple(pd.Series(err["s1_name"])).tolist()
    b = _simple(pd.Series(err["rec_name"])).tolist()
    sim = process.cpdist(a, b, scorer=fuzz.ratio, workers=-1)
    tags = []
    own = owner[err.rec.to_numpy()]
    for k, ra, rn, d, sm, o in zip(err.kind, err.rec_address, err.rec_name, dup[err.s1], sim, own):
        t = ["no_addr" if not str(ra).strip() else "addr"]
        t.append("native" if _RE_INDIC.search(str(rn)) else "latin")
        t.append("name_shared" if d > 1 else "name_unique")
        t.append("sim<50" if sm < 50 else ("sim50-85" if sm < 85 else "sim>=85"))
        if k == "FP":
            t.append("distractor" if o < 0 else "other_owner")
        tags.append(" ".join(t))
    err["tags"] = tags
    tab = (err.groupby(["kind", "tags"])["pts"].agg(["size", "sum"]).rename(columns={"size": "pairs", "sum": "pts"}))
    tab["pts"] = 100 * tab["pts"] / E
    tab = tab.sort_values("pts", ascending=False)
    LOG.info("Loss report by error type (pts = macro-F0.5 points x100 recoverable):\n%s",
             tab.head(25).round(3).to_string())
    for k in ("FP", "FN", "MISS"):
        d = err[err.kind == k].sort_values("pts", ascending=False).head(n_show)
        if len(d):
            with pd.option_context("display.width", 250, "display.max_colwidth", 45):
                LOG.info("Examples %s:\n%s", k, d[["p", "tags", "s1_name", "s1_address", "rec_name", "rec_address"]]
                         .round(3).to_string(index=False))
    err.to_csv(os.path.join(model_dir, "loss_report.tsv"), sep="\t", index=False)
