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


def test_transliteration_and_legal_forms():
    from src.normalize import _prep_one, clean
    assert clean("आदित्य") == "aditya"
    assert clean("महाराष्ट्र") == "maharashtra"
    assert clean("Fractales Amis Groupe S.A.S") == "fractales amis groupe sas"
    assert clean("wilfordhancock.com") == "wilfordhancock"
    # Devanagari and Latin renderings of the same Indian name meet after normalisation
    assert _prep_one("श्री महालक्ष्मी ट्रेडर्स प्राइवेट लिमिटेड", "", "India")[1] == \
        _prep_one("Shree Mahalaxmi Tredars Pvt. Ltd.", "", "India")[1]
    # legal forms are removed wherever they appear
    assert _prep_one("LLC Moncada Learning Center", "", "US")[1] == "moncada learning center"
    assert _prep_one("Zephay Labs SARL", "", "France")[1] == "zephay laboratories"


def test_write_id_lists_format(tmp_path):
    from src.utils import MATCHING_HEADER, write_id_lists
    p = tmp_path / "m.tsv"
    write_id_lists(str(p), MATCHING_HEADER, ["S1-1", "S1-2", "S1-3"], np.array([2, 0, 0]),
                   np.array(["S3-9", "S2-5", "S2-5"], dtype=object), order=np.array([0.9, 0.2, 0.8]))
    lines = p.read_text().splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1:] == ["S1-1\tS2-5", "S1-2\t", "S1-3\tS3-9"]     # every S1 row, no duplicate ids


def test_group_context():
    from src.features import _group_context
    g = np.array([5, 5, 5, 7])
    x = np.array([0.2, 0.9, 0.5, 0.3], dtype=np.float32)
    rank, gap = _group_context(g, x)
    assert rank.tolist() == [3, 1, 2, 1]
    assert np.allclose(gap[:3], [0.2 - 0.9, 0.9 - 0.5, 0.5 - 0.9])
    assert np.isnan(gap[3])                                        # lone candidate: no competitor


def test_stage2_group_stats():
    from src.stage2 import GroupStats
    g = np.array([0, 0, 1, 0])
    p = np.array([0.2, 0.9, 0.7, 0.6], dtype=np.float32)
    st = GroupStats(g, p, 3)
    assert st.rank.tolist() == [3, 1, 1, 2]
    assert np.isclose(st.top1[0], 0.9) and np.isclose(st.top2[0], 0.6) and np.isnan(st.top2[1])
    assert st.n.tolist() == [3, 1, 0] and st.c5.tolist() == [2, 1, 0]


def test_address_normalisation_real_patterns():
    from src.normalize import _prep_one, address_tokens
    assert address_tokens("531 FIFTEENTH AVE, PO BOX 4442, LONGVIEW, WA", "US") == \
        address_tokens("531 15th Avenue, Longview, WA", "US")
    assert "2839" in address_tokens("OH, 002839 SEVENTH ST", "US")
    assert _prep_one("5tar Brothers", "", "India")[1] == "star brothers"
    assert _prep_one("1st Choice Bakery", "", "US")[1] == "1st choice bakery"
    assert _prep_one("राज बिल्डर्स प्रा. लि.", "", "India")[1] == "raj bildars"
