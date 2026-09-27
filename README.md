# BER: Business Entity Resolution (Amazon ML Challenge 2026)

For every Source 1 business, this pipeline finds the matching Source 2 / Source 3 records. It is
built for the full competition data (~2M Source 1 entities and ~10M Source 2/3 records per split)
on a Kaggle notebook (4 CPU cores, ~30 GB RAM; the cross-encoder wants the T4 x2 GPU accelerator).

Stages:
1. normalise (Indic-script transliteration plus a transliteration lexicon learned from the training
   ground truth);
2. scalable blocking (FAISS kNN on name, name+address and address + sparse key index, per country,
   in record chunks);
3. gradient-boosted pair scoring (XGBoost on GPU, LightGBM on CPU);
4. stage 2: competitor-aware re-scoring, optionally with a character-level transformer
   cross-encoder as an extra signal (trained from scratch, GPU);
5. a decision layer tuned for macro F0.5.

Training mimics the test split: the test has ~5.75 Source 2/3 records per Source 1 entity against
~4.68 in train (same matches per entity, so the extra records are distractors). Training therefore
drops ~19% of its Source 1 entities, and their records become distractors (`--s1-drop`, sized
automatically from the test split's row counts).

- Strategy: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Methodology write-up: [`Documentation_template.md`](Documentation_template.md)

**Kaggle (GPU):** [`notebooks/kaggle_gpu.ipynb`](notebooks/kaggle_gpu.ipynb). Its cells clone `main`, install the extra packages, locate the dataset, train, run inference, validate and package the
submission.

**How high can the score go?** On the training split, 39.6 % of Source 1 names are shared by more
than one entity, and 4.4 % of matched records have no address. A record with no address whose
owner's name belongs to several entities cannot be attributed from its content. Those records
alone cap macro F0.5 at about **0.995** for any method (exact name copies alone cost 0.0011); see
`scripts/upper_bound.py`.

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
- `--s1-drop FRAC`: fraction of training Source 1 entities turned into distractors (default: auto from
  `--test-dir`, which defaults to `../test` next to `--data-dir`; `0` = off).
- `--cross-encoder auto|on|off`: transformer cross-encoder stacked into stage 2 (`auto` = only when a
  CUDA GPU is visible). It is kept only if it beats stage 2 without it on nested CV.
  `--ce-epochs`, `--ce-max-train-pairs` size its training.
- `--no-lexicon`: skip the learned transliteration lexicon.

Training ends with a **loss report**, which is also saved as `models/loss_report.tsv`. It gives the
macro-F0.5 points lost to each error type (false positives on distractors vs other owners, scored
false negatives, blocking misses), tagged by record without address, shared Source 1 name, native
script and name similarity, with examples.

Error analysis and experiment log:
- `python scripts/error_analysis.py --data-dir $DATA/train --model-dir models` buckets the worst OOF
  false positives / negatives (including blocking misses) into noise / edge-case / sparse / boundary.
- [`experiments.csv`](experiments.csv) records every ablation with its CV score and whether it was
  kept or reverted.

Try the whole pipeline on generated data:

```bash
python scripts/make_synthetic_data.py --out data/dataset --hard   # --hard: real-data noise + test distractors
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

No pretrained models are used (the cross-encoder is trained from scratch on the training split), and no
external data or services are called. PyTorch (BSD) is used for the cross-encoder.
