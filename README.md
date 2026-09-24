# BER: Business Entity Resolution (Amazon ML Challenge 2026)

A pipeline that links each Source 1 reference business to its Source 2 / Source 3 records. It runs
in three stages: high-recall blocking, LightGBM pair scoring, and a decision layer tuned for macro
F0.5.

- Strategy and architecture: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Methodology write-up: [`Documentation_template.md`](Documentation_template.md)

## Quick start

```bash
pip install -r requirements.txt

# put the competition files here (names are auto-detected: *source*1*, *source*2*, *source*3*, *ground*truth*)
#   data/train/{source1,source2,source3}.tsv  data/train/train_ground_truth.tsv
#   data/test/{source1,source2,source3}.tsv
./run_pipeline.sh

# or try it end-to-end on a generated dataset (includes a labelled hold-out)
SYNTHETIC=1 ./run_pipeline.sh
```

You can also run each step on its own:

```bash
python -m src.train     --data-dir data/train --model-dir models
python -m src.inference --data-dir data/test  --model-dir models --out-dir output [--gt path/to/labels.tsv] [--dump-scores]
python -m src.blocking  --data-dir data/test  --out output/candidate_pairs.tsv     # blocking only
python -m pytest -q tests
```

Paths can be overridden with `--s1 --s2 --s3 --gt`.

Column detection:
- **Id column:** the column containing `id`.
- **Name column:** the column containing `name`.
- **Address:** every remaining address-like column, concatenated.

## Layout

```
src/config.py       all hyper-parameters (blocking budget, LightGBM, decision grid)
src/utils.py        TSV I/O, exact macro-F0.5 metric (+ vectorised version for threshold search)
src/normalize.py    text normalisation, abbreviation/legal-suffix handling, address parsing, phonetics
src/blocking.py     6 candidate generators (TF-IDF kNN x2, token index, phonetic, address, MinHash LSH)
src/features.py     107 vectorised pair features (rapidfuzz cpdist, sparse ops, context features)
src/postprocess.py  exclusivity + expected-F0.5 subset selection, OOF rule search
src/pipeline.py     shared load -> normalise -> block -> featurise (train/inference parity)
src/train.py        entity-grouped stratified CV, OOF, feature importance, nested threshold tuning
src/inference.py    produces output/candidate_pairs.tsv and output/matching_results.tsv
scripts/make_synthetic_data.py   realistic noisy dataset generator for local testing
tests/test_core.py  metric edge cases, fast/exact metric agreement, decision logic
```

## Outputs

| File | Format |
|---|---|
| `output/matching_results.tsv` | `source1_id<TAB>matches`: one row per Source 1 entity, comma-separated ids, empty string for no match. The header is copied from the training ground truth. |
| `output/candidate_pairs.tsv` | `source1_id<TAB>candidate_id<TAB>candidate_source`: every pair produced by blocking. |
| `models/cv_report.json` | Blocking recall and F0.5 ceiling, OOF logloss/AUC, tuned rule, nested CV score. |
| `models/feature_importance.csv`, `models/threshold_curve.csv`, `models/oof_predictions.tsv` | Material for error analysis. |

## Results on the synthetic benchmark

These numbers come from 6k S1 entities and ~8.8k noisy records in train, plus a 4k-entity hold-out.

| Metric | Value |
|---|---|
| Blocking pair recall (train / test) | 0.994 / 0.999 |
| Nested-CV macro F0.5 | 0.970 |
| Hold-out macro F0.5 | 0.968 |
| Full run time (4 CPU cores) | ~1.5 min |
