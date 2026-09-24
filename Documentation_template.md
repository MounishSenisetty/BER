# Methodology: Business Entity Resolution

> If the organisers' official template uses different headings, map these sections onto them.
> Numbers marked *(synthetic)* come from `scripts/make_synthetic_data.py`. Replace them with the
> figures from `models/cv_report.json` after training on the competition data.

## 1. Problem understanding

For every Source 1 (deduplicated reference) business we must return every Source 2 / Source 3
record that refers to it. The scoring is macro F0.5 over Source 1 entities, where precision
counts twice as much as recall. An entity with no true match scores 1 only if we predict nothing.
Only names and addresses are available, and no external data or services are used.

Two structural facts shape the solution:

- Every S2/S3 record belongs to at most one S1 entity, because S1 is deduplicated.
- The metric rewards per-entity decisions rather than a single global cut-off.

## 2. Pipeline overview

`normalise → block (6 generators, capped) → 107 pair features → LightGBM (5 entity-grouped folds,
averaged) → exclusivity + expected-F0.5 subset selection → matching_results.tsv`

| Stage | Code | Output |
|---|---|---|
| Normalisation & address parsing | `src/normalize.py` | per-record canonical strings, tokens, phonetic keys, house/postal/unit |
| Blocking | `src/blocking.py` | `candidate_pairs.tsv` |
| Features | `src/features.py` | pair feature matrix |
| Training & CV | `src/train.py` | `models/model_bundle.pkl`, CV report |
| Decision layer | `src/postprocess.py` | per-entity match sets |
| Inference | `src/inference.py` | `matching_results.tsv` |

## 3. Pre-processing

Each record goes through the following steps:

- Unicode folding, lowercasing, and `&`→`and`; apostrophes and punctuation are removed.
- Glued alphanumerics are split (`suite12` → `suite 12`).
- Abbreviations are canonicalised for business words, street types and directions.
- Legal suffixes are stripped to form the *core name*.
- Addresses are parsed into house number, postal code (last 5–6 digit number), unit/suite and first
  street token.
- Phonetic codes (metaphone, soundex) are computed for the first token, plus a sorted-metaphone
  signature of the whole name.

Source files with split address columns (street/city/state/zip) are concatenated automatically.

## 4. Blocking (candidate generation)

Each S2/S3 record queries the S1 index. The candidate set is the union of six generators:

1. Char 2–4-gram TF-IDF kNN on the core name (top 25).
2. The same on name + address (top 25).
3. Rare-name-token inverted index (tokens in ≤100 S1 rows).
4. Phonetic compound keys: metaphone/soundex of the first token × postal code or house number.
5. Address compound keys: house number × street, and postal code × first name token.
6. MinHash LSH (datasketch, 64 permutations, Jaccard threshold 0.35) over name+address tokens.

The union is re-ranked with a cheap similarity score and capped at 40 candidates per record. The
blocking audit reports pair recall, the macro-F0.5 ceiling (the score an oracle classifier could
reach on these candidates) and the reduction ratio.

*(synthetic)* Pair recall is 0.994 on train and 0.999 on test. The F0.5 ceiling is 0.998. The
reduction ratio versus the full cross product is 99.3%.

## 5. Features (107)

| Group | Examples |
|---|---|
| Fuzzy name | Jaro-Winkler, Levenshtein ratio, partial ratio, token sort/set ratio, compact-string JW, first-token JW |
| Vector | char-TF-IDF cosine (name, address, full), word-TF-IDF cosine (name, address) |
| Set overlap | IDF-weighted Jaccard, directional containment, unexplained IDF mass per side (name, address, full) |
| Address atoms | house / postal / unit / street equality (NaN when missing), numeric-token agreement in address and name |
| Phonetic | metaphone / soundex equality, phonetic-signature ratio, acronym match |
| Frequency | log frequency of the core name and first token (chain detection) |
| Provenance | blocker flags, kNN ranks, cheap score / rank, record source (S2 vs S3) |
| Context | rank and margin to the best *competing* candidate within the record and within the S1 entity |

All features are computed as vectorised array operations: rapidfuzz `cpdist` (multi-threaded) and
sparse row-wise products.

## 6. Model

- LightGBM binary classifier: learning rate 0.03, 63 leaves, feature/bagging fraction 0.8.
  Training uses early stopping on the validation fold's logloss.
- The five fold models are averaged at inference time. This keeps test probabilities on the same
  scale as the OOF probabilities that the decision threshold was tuned on.

*(synthetic)* OOF logloss is 0.0040 and AUC is 0.9998. The top features by gain are the
record-level margin to the best competitor, the cheap-score rank and the name+address cosine.

## 7. Decision layer for macro F0.5

1. **Exclusivity:** a record may only be assigned to its highest-probability S1 entity.
2. **Expected-F0.5 subset selection per entity.** Candidates are sorted by p, and the top-k is
   chosen to maximise `E[F|k] ≈ 1.25·Σ_{i≤k} p_i / (0.25·Σ p_i + k)`. The empty prediction's value
   is compared against `E[F|0] = Π(1 − p_i)`, which is the probability that the entity is a
   singleton.
3. The mode (expected-F vs global threshold), the exclusivity switch and a probability floor are
   grid-searched on OOF predictions. The objective includes entities whose matches were missed by
   blocking. The final threshold is the median of the near-optimal plateau.

*(synthetic)* The selected rule is expected-F with exclusivity on and a floor of 0.30.

## 8. Validation

- Five folds are assigned **per Source 1 entity** and stratified by match count (0/1/2/3+). Every
  candidate pair inherits its entity's fold, so match sets are never split across folds.
- Decision-rule tuning is nested: tune on four folds' OOF, evaluate on the fifth.

| Estimate | Macro F0.5 *(synthetic)* |
|---|---|
| OOF, rule tuned on all folds | 0.9706 (singletons 0.976 / matched 0.968) |
| Nested CV | 0.9700 (folds 0.965–0.974) |
| Independent hold-out split | 0.9678 |

## 9. Reproducibility

```bash
pip install -r requirements.txt
./run_pipeline.sh                    # train on data/train, predict data/test -> output/
SYNTHETIC=1 ./run_pipeline.sh        # self-contained demo on generated data
python -m pytest -q tests            # metric / decision-layer unit tests
```

- Seeds are fixed, so a CPU-only run is deterministic.
- A full synthetic run takes about 1.5 minutes on 4 cores.
- All settings live in `src/config.py`.

## 10. Compliance

- Only the provided tables are used.
- There are no geocoding calls, external APIs, reference databases or downloaded models.
- TF-IDF statistics are fitted on each split's own text and use no labels.
