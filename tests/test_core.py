import numpy as np
import pandas as pd
import pytest

from src.postprocess import rec_argmax_mask, select
from src.utils import entity_f05, fast_macro_f05, load_ground_truth, macro_f05


def test_entity_f05_edge_cases():
    assert entity_f05(set(), set()) == 1.0                    # singleton, predicted empty
    assert entity_f05(set(), {"a"}) == 0.0                    # singleton, any FP
    assert entity_f05({"a"}, set()) == 0.0                    # missed everything
    assert entity_f05({"a"}, {"b"}) == 0.0
    assert entity_f05({"a", "b"}, {"a", "b"}) == 1.0
    # P = 1, R = 0.5 -> 1.25*0.5/(0.25+0.5) = 0.8333
    assert entity_f05({"a", "b"}, {"a"}) == pytest.approx(1.25 * 0.5 / 0.75)
    # P = 0.5, R = 1 -> 1.25*0.5/(0.125+1) = 0.5556  (precision weighs more)
    assert entity_f05({"a"}, {"a", "b"}) == pytest.approx(1.25 * 0.5 / 1.125)


def test_macro_missing_entities_count_as_empty():
    truth = {"e1": set(), "e2": {"x"}}
    assert macro_f05(truth, {}) == 0.5
    assert macro_f05(truth, {"e2": ["x"]}) == 1.0


def test_fast_metric_matches_exact():
    rng = np.random.default_rng(0)
    E, n = 200, 2000
    ent = rng.integers(0, E, n)
    label = rng.random(n) < 0.2
    sel = rng.random(n) < 0.3
    missed = rng.integers(0, 2, E)                           # true matches blocking never proposed
    n_true = np.bincount(ent, weights=label, minlength=E) + missed
    truth = {e: {f"p{i}" for i in np.flatnonzero((ent == e) & label)} | {f"miss{e}_{j}" for j in range(missed[e])}
             for e in range(E)}
    pred = {e: {f"p{i}" for i in np.flatnonzero((ent == e) & sel)} for e in range(E)}
    assert fast_macro_f05(ent, sel, label, n_true) == pytest.approx(macro_f05(truth, pred))


def test_exclusivity_keeps_one_owner_per_record():
    rec = np.array([0, 0, 1, 1, 1])
    p = np.array([0.2, 0.9, 0.5, 0.7, 0.1])
    assert rec_argmax_mask(rec, p).tolist() == [False, True, False, True, False]


def test_expected_f_selection():
    ent = np.array([0, 0, 1, 2, 2])
    rec = np.arange(5)
    # entity 0: one sure match + one weak -> keep only the sure one
    # entity 1: lone weak candidate -> P(singleton)=0.8 beats predicting it
    # entity 2: two strong candidates -> keep both
    p = np.array([0.99, 0.30, 0.20, 0.95, 0.90])
    sel = select(ent, rec, p, "expected_f", 0.0, exclusive=False)
    assert sel.tolist() == [True, False, False, True, True]
    assert select(ent, rec, p, "threshold", 0.5, exclusive=False).tolist() == [True, False, False, True, True]


def test_ground_truth_parsing(tmp_path):
    f = tmp_path / "gt.tsv"
    f.write_text("source1_id\tmatches\nA\tx,y\nB\t\n")
    truth, header = load_ground_truth(str(f), ["A", "B", "C"])
    assert header == ("source1_id", "matches")
    assert truth == {"A": {"x", "y"}, "B": set(), "C": set()}
