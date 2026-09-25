# Methodology: Business Entity Resolution

> Copy these sections into the official `Documentation_template.md` headings. Values marked
> **[fill]** come from `models/cv_report.json` and the inference log of your run.

## 1. Problem understanding

Source 1 is a deduplicated reference set. For each Source 1 entity we return every Source 2 /
Source 3 record that refers to the same business. The score is macro F0.5 per Source 1 entity:

- precision is weighted twice as much as recall;
- a singleton scores 1 only when we predict nothing.

Only the provided names, addresses and country labels are used. There are no external lookups,
APIs, geocoders or pretrained models.

The data has a few properties that shape the solution:

- **Scale.** About 2.2M / 10.3M (train) and 1.7M / 10.0M (test) Source 1 / Source 2+3 records.
  Every stage streams country partitions and record chunks so it fits a 4-core, 30 GB machine.
- **Countries.** Train has US and India; test adds France. The country label is used only to
  partition blocking and to choose abbreviation tables. It is never a model feature, so the model
  transfers to France.
- **Noise.** Legal forms are dropped, changed or moved to the front. Names and addresses may be
  in Devanagari or Kannada script. There are typos, website-style names, garbage prefixes,
  reordered or missing address components, landmark references and `null` tokens.

## 2. Pipeline

`normalise → per-country index (TF-IDF/SVD/FAISS + key index) → capped candidates → ~100
features → LightGBM (5 entity-grouped folds, averaged) → expected-F0.5 decision → outputs`

## 3. Normalisation

- **Rule-based Indic transliteration.** Covers Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
  Tamil, Telugu, Kannada and Malayalam, including Hindi final-schwa deletion. This runs before
  accent folding, because Indic vowel signs are combining marks.
- **Clean-up.** Dotted acronyms are glued (`S.A.S` → `sas`, `L.L.C.` → `llc`). Web wrappers are
  stripped (`wilfordhancock.com` → `wilfordhancock`). Then NFKD accent folding, lowercasing and
  punctuation removal.
- **Legal forms** for US, India (including transliterated `प्राइवेट लिमिटेड`) and France
  (SARL, SAS, SCI, EURL…) are removed anywhere in the name, giving the *core name*.
- **Abbreviations** are canonicalised with a common table plus country tables. For example, the
  France table maps `St` → `saint` while the US table maps `Ste` → `suite`. India has its own
  words: `nr` → near, `opp`, `sector`, `nagar`, and city aliases such as Bengaluru → Bangalore.
- **State names → codes** for US and India, including transliterated native-script names.
- **Indian name spelling.** `ph→f`, `oo→u`, `ee→i`, `x→ksh`, `w→v`, and doubled letters are
  collapsed. This makes Latin and Devanagari renderings of the same name meet, e.g.
  `Shree Mahalaxmi` / `श्री महालक्ष्मी` → `shri mahalakshmi`.
- **Address parsing.** House number, postal code (5–6 digits), unit number and first street
  token.

## 4. Candidate generation (blocking)

For each country, the Source 1 side is indexed once. Each chunk of 400k Source 2/3 records then
queries that index:

1. **Dense kNN on the core name.** Hashed char 3-gram TF-IDF (idf fitted on Source 1) is
   compressed with TruncatedSVD to 128 dimensions, then searched with FAISS IVF inner product.
   Top 10.
2. **The same on name + address.** Top 10. This separates chains and generic names by location.
3. **Sparse key index.**
   - Keys: rare name tokens, token pairs and the compact name.
   - Compound keys: house number × street, postal code × first token, metaphone × postal code,
     metaphone × house number, street × first token.
   - Keys shared by more than 100 Source 1 rows are dropped. Keys are idf-weighted, and matching
     is a sparse matrix product. Top 10.

The three lists are merged and ranked by a cheap score (dense cosines plus key overlap), and the
top 10 per record are kept. This capped set is exactly what the model scores and what
`candidate_pairs.tsv` contains.

Results on train: pair recall **[fill]**; F0.5 ceiling **[fill]**; recall by cap
(`recall_at_cap`) **[fill]**; about **[fill]** candidates per record.

## 5. Features (~100)

| Group | Features |
|---|---|
| Fuzzy name | Jaro-Winkler, Levenshtein ratio, partial ratio, token sort/set, compact-string JW, first-token JW |
| Vector | exact char-TF-IDF cosine (name, address, full), SVD cosines |
| Set overlap | idf-weighted Jaccard, containment both ways, unexplained idf mass (name, address, full) |
| Address atoms | house / postal / unit / street equality (missing ≠ different), numeric-token agreement |
| Phonetic | metaphone / soundex, phonetic-signature ratio, acronym |
| Frequency | how common the core name is among the country's Source 1 (chains) |
| Provenance | which generator proposed the pair, ranks, key score, cheap score |
| Context | rank and margin of the pair among the record's other candidates (soft exclusivity: a record has at most one owner) |

## 6. Model and training

- **Blocking runs over the full training split**, so candidate density and competition between
  similar entities match test.
- **Stratified sample of 120k Source 1 entities**, stratified by match count. Every candidate
  pair of a sampled entity is labelled and featurised.
- **Classifier:** LightGBM binary (logloss), 5 folds grouped by Source 1 entity, early stopping.
  The fold models are averaged at inference.
- **Results:** OOF logloss **[fill]**, AUC **[fill]**.

## 7. Decision layer (macro F0.5)

For each entity, candidates are sorted by probability p. We compute:

- `E[F|k] ≈ 1.25·Σ_{i≤k} p_i / (0.25·Σ p_i + k)`, the expected score of predicting the top k;
- `E[F|0] = Π(1 − p_i)`, the probability that the entity is a singleton.

We then predict the best k, or nothing. This is compared with a global threshold, and the mode and
floor are grid-searched on OOF predictions over all sampled entities, including their blocking
misses. A nested estimate tunes on 4 folds and evaluates on the 5th.

Selected rule **[fill]**. Nested-CV macro F0.5 **[fill]** (singletons **[fill]**, matched
**[fill]**).

## 8. Validation and reproducibility

- Folds are split per Source 1 entity, stratified by match count 0/1/2/3+. Entities are never
  split across folds.
- The outputs pass `utils/validate_submission.py` (with `--check-ids`).
- Run commands are in `README.md`. Seeds are fixed. Runtime on Kaggle CPU is **[fill]**.
