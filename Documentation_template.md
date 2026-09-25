# ML Challenge 2026: Business Entity Resolution (Methodology)

## 1. Summary

For every Source 1 (reference) business, we return all Source 2 / Source 3 records that describe
the same real-world business. The pipeline has four stages:

1. **Normalisation.** Transliteration of Indic scripts, legal-form removal, abbreviation, state
   and city canonicalisation, and address parsing.
2. **Candidate generation.** Scalable, per-country blocking: FAISS nearest-neighbour search on
   SVD-compressed character n-gram TF-IDF, plus an idf-weighted sparse key index.
3. **Pair classification.** A gradient-boosted tree model (XGBoost, 5 entity-grouped folds)
   scores each candidate pair using 103 engineered features.
4. **Decision layer.** It picks each entity's match set by maximising the *expected* per-entity
   F0.5, which also decides explicitly when to predict "no match".

| Result (training data, entity-grouped cross-validation) | Value |
|---|---|
| Candidate recall (true pairs present in `candidate_pairs`) | **97.25 %** |
| Macro-F0.5 ceiling given the candidates | **0.9905** |
| **Nested-CV macro F0.5** (threshold tuned on 4 folds, scored on the 5th) | **0.9681** (per fold 0.9671 – 0.9686) |
| OOF macro F0.5, singletons / matched entities | 0.951 / 0.969 |
| Public leaderboard F0.5 | *[fill in after submission]* |

Only the provided training/test files are used. There are no external databases, APIs, geocoders or
pretrained models. The models are XGBoost (Apache-2.0), with LightGBM (MIT) as a CPU fallback.

---

## 2. Data understanding

| | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| Train rows | 2,206,821 | 5,034,616 | 5,285,603 |
| Test rows | 1,732,544 | 4,887,273 | 5,082,316 |
| Train countries | US 60 %, India 40 % | same | same |
| Test countries | India 810k, US 663k, **France 259k** | same | same |

Findings from our exploratory analysis (`scripts/eda.py`) that shaped the design:

- **Ground truth.** There are 7,638,365 true pairs. Only **5.6 %** of Source 1 entities are
  singletons; the mean is **3.46 matches per entity** (maximum 11). 73–75 % of Source 2/3 records
  are matched; the rest are distractors.
- **One owner per record.** No Source 2/3 record matches more than one Source 1 entity, which
  confirms that Source 1 is deduplicated. We use this as the basis for "competition" features
  between candidates.
- **No cross-country matches.** Every true pair shares its country label, so blocking is
  partitioned by country. The label is treated as an open set, and France (test only) gets its
  own partition.
- **Scripts.** 24 % of India Source 2 names (13 % of Source 3) are written in Devanagari,
  Kannada, Tamil or Gujarati script. Many addresses mix scripts (`…, महाराष्ट्र`).
- **Name noise.** About 10 % of true matches have almost no name overlap. Name token-set
  similarity at the 10th percentile is 59 for India and 78 for the US. They fall into two
  groups:
  - *unrelated trade names* (`Olanay Altisource LLC` ↔ `Umbramiraquo`), where only the address
    links the records;
  - *acronyms* (`Wilson Express Canada Inc` ↔ `WEC`).
- **Other noise.**
  - Legal forms dropped, abbreviated, misspelt or moved to the front (`Privhea`, `Liimted`,
    `(Limitend)`, `[LLP] Swasthya Global`).
  - Honorifics (`Smt`, `M/s`) and digit-for-letter typos (`H0ly`, `5tar`, `6old`).
  - Website-style names (`quogild.com`) and garbage prefixes (`--`, `<<`).
  - Literal `null` inside addresses, reordered address components, landmark references
    (`Near SBI ATM`, `Opp.`), and state names vs. codes vs. native script (`Karnataka` / `KA` /
    `ಕರ್ನಾಟಕ`).
- **Chains / generic names.** 8–11 % of core names are shared by more than one Source 1 entity
  (e.g. `meridian` ×77 in the US). For these, the address is decisive.

---

## 3. Methodology

### 3.1 Normalisation (`src/normalize.py`)

Each record is normalised once, in parallel processes, into the fields used by every later stage.

1. **Indic transliteration.** A rule table covers the ISCII-aligned Unicode blocks (Devanagari,
   Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam). It handles consonants,
   vowel signs, virama and nukta, and applies Hindi final-schwa deletion except after conjuncts
   ending in *ya/ra/va* (`आदित्य` → `aditya`, `महाराष्ट्र` → `maharashtra`). This runs **before**
   accent folding, because Indic vowel signs are Unicode combining marks. Doubled vowels are
   collapsed only inside transliterated runs.
2. **Cleaning.**
   - Glue dotted acronyms (`S.A.S` → `sas`, `L.L.C.` → `llc`) and strip web wrappers
     (`wilfordhancock.com` → `wilfordhancock`).
   - NFKD accent folding (French accents), lowercasing, `&` → `and`, punctuation → space.
   - Split glued alphanumerics, and fix digit-for-letter typos inside words (`h0ly` → `holy`).
3. **Abbreviations.**
   - A common table (street types, directions, business words), plus country tables.
   - India: `nr` → near, `opp`, `nagar`, `sector`, city aliases Bengaluru/Bangalore,
     Gurugram/Gurgaon, Mumbai/Bombay…
   - France: `r` → rue, `bd` → boulevard, `st` → saint.
   - US and India state names, including transliterated native-script names, are mapped to
     codes.
4. **Legal forms and honorifics** are removed anywhere in the name to form the *core name*:
   - US: Inc, LLC, Corp…;
   - India: Pvt, Private, Limited, LLP, OPC and transliterated `प्राइवेट लिमिटेड`;
   - France: SARL, SAS, SCI, EURL…;
   - misspellings are caught by edit distance ≤ 1–2 against long legal words, plus prefix rules;
   - a short exclusion list keeps real words such as "Privacy", "Limitless" and "Corporate".
5. **Indian spelling normalisation** of names. `ph→f`, `oo→u`, `ee→i`, `x→ksh`, `w→v` and
   repeated letters collapsed, so that `Shree Mahalaxmi` and `श्री महालक्ष्मी` both become
   `shri mahalakshmi`.
6. **Address parsing.** Tokens with `null` dropped; house number, postal code (last 5–6-digit
   number), unit/suite number and first street token; all numeric tokens kept.
7. **Derived keys.** Compact name (spaces removed), initials (acronym), metaphone/soundex of the
   first token, and a sorted-metaphone signature of the name.

### 3.2 Candidate generation / blocking strategy (`src/blocking.py`, `src/pipeline.py`)

**Design goals:**
- find ~98 % of true pairs;
- keep about 11 candidates per record;
- fit 12M records per split into a 4-core, ~30 GB machine.

**Streaming.** The data is processed **country by country**. Each country's Source 1 is indexed
once, and Source 2/3 records are streamed through it in chunks of 300,000. Peak memory is one
country index plus one chunk.

**Source 1 index (per country):**

- **Hashed char-3-gram TF-IDF spaces** for the core name, the address, and name + address. IDF is
  fitted on the country's Source 1, with sublinear tf and L2-normalised rows.
- **Dense vectors.** TruncatedSVD (128 dimensions, fitted on 300k rows) of the name and
  name+address TF-IDF, indexed in **FAISS IVF-Flat inner-product indexes** (nlist ≈ 4√N,
  nprobe 24), which run on the GPU when available.
- **Sparse key index** (hashed to 2²³ columns, idf-weighted; keys held by more than 100
  Source 1 rows are dropped as unselective). Keys cover:
  - *name evidence:* every rare core-name token, sorted pairs of the first four tokens, and the
    compact name;
  - *address evidence:* house number × street token, postal code × first name token, metaphone ×
    postal code, metaphone × house number, street × first name token;
  - *trade-name / acronym evidence:* **adjacent address-token pairs**, and **acronym × address
    number** (for both Source 1 initials and short record names).

**Query (per record chunk).** Three generators run, each returning ~10–15 candidates per record:

1. FAISS top 10 on the core-name vector;
2. FAISS top 10 on the name+address vector;
3. top 15 by key overlap, a sparse matrix product with the key index.

The union is scored with a cheap similarity:
`0.30·cos_name + 0.30·cos_full + 0.25·cos_address + 0.15·tanh(key_score/10)`.

For each record we keep:
- the **top 10 by cheap score**, plus
- the **2 best by address cosine** (if ≥ 0.5), plus
- the **2 best by key overlap**.

These extra "specialist" slots stop trade-name and acronym records from being cut by a
name-dominated ranking. The final set is exactly what the classifier scores and what
`candidate_pairs.tsv` contains, so every match is also a candidate.

**Blocking quality (full training split):**

| Metric | Value |
|---|---|
| Candidate pairs / records | 112,198,941 / 10,320,219 (**10.87 per record**) |
| Reduction ratio vs. within-country cross product (1.18 × 10¹³ pairs) | **99.9990 %** |
| Pair recall (final candidates) | **97.25 %** |
| Pair recall before the per-record cap | 97.54 % |
| Recall at cap 1 / 3 / 5 / 10 (cheap-score rank) | 94.19 / 96.01 / 96.47 / 96.94 % |
| Macro-F0.5 ceiling (oracle classifier on these candidates) | **0.9905** |

**Error analysis of missed true pairs** (`blocking_misses.tsv`, stratified sample):
- 87 % were never generated and 13 % were generated but capped.
- The remaining misses are mostly records with **empty addresses** plus modified names
  (`Veex LLC Service`, `Hill Partners`), and heavily truncated addresses with
  native-script names.

### 3.3 Model architecture and feature engineering (`src/features.py`, `src/model.py`, `src/train.py`)

**Features (103 per pair).** All are computed vectorised per chunk: rapidfuzz `cpdist` in C++
(multi-threaded), sparse row-wise products, and numpy. IDF statistics come from the country's
Source 1, so values are consistent across chunks and splits.

| Group | Features |
|---|---|
| Fuzzy name | Jaro-Winkler, Levenshtein ratio, partial ratio, token-sort / token-set ratio (core name); ratio on the full name; compact-string JW; first-token JW; record name inside the Source 1 address |
| Vector similarity | exact char-TF-IDF cosine (name, address, full); SVD cosines from blocking |
| Token overlap | IDF-weighted Jaccard, containment in both directions, **unexplained IDF mass** per side, and raw counts, for name, address and full token sets; numeric-token agreement for address and name (store numbers) |
| Address atoms | house number / postal / unit / street equality (NaN when missing, so "missing" ≠ "different") |
| Phonetic / acronym | metaphone and soundex of first token, sorted-metaphone signature ratio, acronym flag |
| Frequency | log frequency of the core name and first token among the country's Source 1 (chain detection) |
| Provenance | which generator proposed the pair, kNN ranks, key score, cheap score and rank, record source (S2/S3), transliteration flags |
| **Record context** | rank of the pair among the record's candidates and **margin to the best competing candidate**, for a combined score, name/full cosines, name token-set ratio, key score and cheap score |

The record-context features encode the one-owner-per-record structure softly and dominate the
model. By gain:
- `combo_gap_rec` 64.1 % and `combo_rank_rec` 12.3 %;
- then `cos_full_c_gap_rec` 4.1 %, `num_addr_only2` 3.5 %, `name_full_ratio` 1.5 %,
  `num_addr_only1` 1.2 %, `house_eq` 0.6 %.

The country label is **not** a feature, so the model transfers to France.

**Training data.**
- Blocking runs over the **full** training split, so candidate density and competition between
  similar entities match the test set.
- A stratified sample of **120,000 Source 1 entities** (by match count 0/1/2/3+) is featurised
  with all of their candidate pairs.
- Every candidate of each record touching a sampled entity is featurised, so context features
  see the complete competition. That gives 6,146,574 labelled pairs (6.57 % positive).

**Model.**
- XGBoost binary classifier on GPU (`hist`, lossguide, 127 leaves, η = 0.05, subsample and
  colsample 0.8, max_bin 127), trained with early stopping on each validation fold.
- **5 folds grouped by Source 1 entity** (stratified by match count), so an entity's candidate
  set is never split across folds. Best iterations: 1093–1294.
- Test predictions are the average of the 5 fold models.
- OOF logloss **0.00819**, AUC **0.99982**.

### 3.4 Decision layer for macro F0.5 (`src/postprocess.py`)

Because the score is a per-entity F0.5 average, the decision is made per entity rather than with
one global threshold. For entity *e*, with candidates sorted by predicted probability
p₁ ≥ p₂ ≥ …:

- E[F0.5 | predict top k] ≈ 1.25 · Σ_{i≤k} pᵢ / (0.25 · Σᵢ pᵢ + k)
- E[F0.5 | predict nothing] = Πᵢ (1 − pᵢ), the model's probability that *e* is a singleton.

We predict the k that maximises the expected score (possibly k = 0), considering only candidates
above a probability floor.

The mode (expected-F vs. global threshold) and the floor are grid-searched on OOF predictions.
The search covers **all** sampled entities, including true pairs that blocking missed, so recall
loss is priced in. The chosen floor is the median of the near-optimal plateau.

- **Selected rule:** expected-F with floor **0.64**; OOF macro F0.5 0.9682. The best global
  threshold (0.71) scores 0.9678.
- Singletons score 0.951 and matched entities 0.969.

### 3.5 Validation

- **Folds** are grouped by Source 1 entity and stratified by match count. No entity's
  candidates are split across folds, and no labels are used in blocking or features.
- **Nested evaluation.** The decision rule is tuned on 4 folds' OOF predictions and evaluated on
  the held-out fold: **0.9681** (0.9686, 0.9679, 0.9684, 0.9671, 0.9686). This is our unbiased
  estimate of test performance.
- **Iteration history** on the real data, with the same validation protocol:

| Version | Candidate recall | F0.5 ceiling | Nested-CV macro F0.5 |
|---|---|---|---|
| v1: name/full FAISS + name keys, LightGBM | 92.22 % | 0.9712 | 0.9493 |
| **v2 (final)**: + address-pair and acronym keys, address/key specialist slots, misspelt-legal-form handling, XGBoost-GPU | **97.25 %** | **0.9905** | **0.9681** |

- **Format.** Outputs pass `utils/validate_submission.py`, including the `--check-ids` option on
  `matching_results.tsv`.

---

## 4. Other relevant information

**Compute (Kaggle, 4 vCPU, ~30 GB RAM, 2× T4):**

| Stage | Time |
|---|---|
| Training: load, normalise, block and featurise 10.3M records | ~83 min |
| Training: XGBoost GPU, 5 folds | ~15 min |
| Training: decision search | ~2 min |
| Inference on the test split (9.97M records, ~108M candidate pairs) | ~3.5–4 h |

Peak memory was 28.1 GB during training.

**Reproducibility.** Seeds are fixed. From `code/business_entity_resolution/`:

```bash
pip install -r requirements.txt
python -m src.train     --data-dir <student_resource>/dataset/train --model-dir models
python -m src.inference --data-dir <student_resource>/dataset/test  --model-dir models --out-dir output
```

The backend is selected automatically: XGBoost on GPU if CUDA is available, otherwise LightGBM
on CPU. FAISS uses the GPU if a GPU build is installed.

**Licences and compliance:**
- XGBoost (Apache-2.0), LightGBM (MIT), FAISS (MIT), rapidfuzz (MIT), scikit-learn / pandas /
  numpy / scipy (BSD).
- No pretrained or external models, and no external data, APIs or geocoding.
- Transliteration and abbreviation tables are hand-written rules in the code.

**Limitations and next steps:**
- Records with an empty address *and* a modified name remain the main blocking miss. A
  record-to-record link between Source 2 and Source 3 (transitivity) could recover some of them.
- Hard one-owner-per-record assignment is currently learned softly through context features.
  Scoring all competitors during training would allow tuning it explicitly.
- Second-stage stacking on OOF probabilities, and seed/fold bagging.
