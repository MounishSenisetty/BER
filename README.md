# BER: Business Entity Resolution (Amazon ML Challenge 2026)

For every Source 1 business, this pipeline finds the matching Source 2 / Source 3 records. It is
built for the full competition data (~2M Source 1 entities and ~10M Source 2/3 records per split)
on a Kaggle CPU notebook.

Stages:
1. normalise (including Indic-script transliteration);
2. scalable blocking (FAISS kNN + sparse key index, per country, in record chunks);
3. LightGBM pair scoring;
4. a decision layer tuned for macro F0.5.

- Strategy: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Methodology write-up: [`Documentation_template.md`](Documentation_template.md)

## Reproduce end-to-end

```bash
pip install -r requirements.txt
DATA=/path/to/student_resource/dataset
python scripts/eda.py --train-dir $DATA/train --test-dir $DATA/test            # optional analysis
python -m src.train     --data-dir $DATA/train --model-dir models             # blocking + features + CV + threshold
python -m src.inference --data-dir $DATA/test  --model-dir models --out-dir output
python $DATA/../utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir $DATA/test
python scripts/make_submission_zip.py --team <team> --out-dir output --doc Documentation_template.md
```

Input files are found automatically (`train_source1.tsv` … `train_ground_truth.tsv`,
`test_source1.tsv` …). The columns are `entity_id, business_name, business_address, country`.

Useful flags:
- `--train-entities N`: Source 1 entities used for training (default 120k; `0` = all).
- `--max-candidates K`: candidates kept per record (default 10).
- `--chunk-records N`: records per chunk; lower it if memory is tight.

Try the whole pipeline on generated data:

```bash
python scripts/make_synthetic_data.py --out data/dataset
python -m src.train --data-dir data/dataset/train --model-dir models --train-entities 0
python -m src.inference --data-dir data/dataset/test --model-dir models --out-dir output \
       --gt data/dataset/test_labels/test_ground_truth.tsv
```

## Outputs

| File | Format |
|---|---|
| `output/matching_results.tsv` | `source1_entity_id<TAB>matched_entity_ids`. One row per test Source 1 entity; ids are comma-separated; the list is empty for no match. |
| `output/candidate_pairs.tsv` | `source1_entity_id<TAB>candidate_entity_ids`. Exactly the capped candidate set the classifier scores, so every match is also a candidate. |
| `models/cv_report.json` | Blocking recall and F0.5 ceiling, recall by per-record cap, OOF logloss/AUC, tuned rule, nested-CV macro F0.5. |

## Layout

```
src/config.py       all settings (blocking budget, FAISS, LightGBM, decision grid)
src/utils.py        TSV I/O, exact macro-F0.5 metric, output writers
src/normalize.py    Indic transliteration, legal forms (US/IN/FR), abbreviations, state/city aliases, address parsing
src/blocking.py     per-country index: hashed char TF-IDF -> SVD -> FAISS kNN (name, name+address) + sparse key index
src/features.py     ~100 vectorised pair features (rapidfuzz cpdist, sparse ops, record-context features)
src/postprocess.py  expected-F0.5 subset selection / threshold, OOF rule search
src/pipeline.py     streaming driver: country partitions x record chunks (bounded memory)
src/train.py        full-split blocking, sampled-entity features, entity-grouped CV, nested threshold tuning
src/inference.py    writes candidate_pairs.tsv + matching_results.tsv
scripts/            eda.py, make_synthetic_data.py, make_submission_zip.py
tests/test_core.py  metric, decision layer, normalisation, output format
```

Licences:
- LightGBM: MIT.
- FAISS: MIT.
- rapidfuzz: MIT.
- scikit-learn: BSD.

No pretrained models are used, and no external data or services are called.
