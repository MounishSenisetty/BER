# Strategic & Architectural Roadmap: Business Entity Resolution (Amazon ML Challenge 2026)

## 0. The three facts that drive every design decision

1. **The metric is macro F0.5 per Source 1 entity.** Precision is weighted 4x recall in the
   denominator: `F0.5 = 1.25·TP / (0.25·|T| + |P|)`. A false positive costs 1 unit and a missed
   match costs 0.25 units. Singletons (≈ a third of entities in typical ER data) score 1.0 or 0.0,
   with nothing in between, so one stray FP wipes out an entity entirely.
2. **Source 1 is deduplicated.** Every S2/S3 record belongs to *at most one* S1 entity. That is a
   hard structural constraint (exclusivity) and a strong feature: how a pair compares with the
   record's other candidates.
3. **Names + addresses only, no external data.** All signal has to come from string similarity,
   token rarity (IDF), parsed address atoms (house number, postal code, unit), and the
   *context* of competing candidates.

---

## 1. Multi-stage pipeline

```
 S1 (reference) ─┐                    ┌─────────────── Stage 1: BLOCKING (recall) ───────────────┐
 S2 (vendor A) ──┼─ normalise/parse ─▶│ char-TFIDF kNN(name) ∪ char-TFIDF kNN(name+addr)          │
 S3 (vendor B) ──┘  (once per record) │ ∪ rare-token index ∪ phonetic keys ∪ address keys ∪ LSH   │
                                      │ → cheap re-rank → cap K per record → candidate_pairs.tsv  │
                                      └──────────────────────────┬────────────────────────────────┘
                                                                 ▼
                      ┌──────────── Stage 2: PAIR SCORING (precision) ────────────┐
                      │ 2A  ~107 features → LightGBM (5 entity-grouped folds)     │
                      │ 2B  (optional) cross-encoder on top-k → stacked feature   │
                      └──────────────────────────┬────────────────────────────────┘
                                                 ▼
                      ┌──────────── Stage 3: DECISION (macro F0.5) ────────────────┐
                      │ exclusivity (record → arg-max S1)                          │
                      │ per-entity expected-F0.5 subset selection OR global cut    │
                      │ rule + threshold tuned on OOF (nested) → matching_results  │
                      └────────────────────────────────────────────────────────────┘
```

### 1.1 High-recall blocking

**Normalisation first (`src/normalize.py`).** Unicode fold, lowercase, `&`→`and`, strip
punctuation, split glued alphanumerics (`suite12` → `suite 12`), canonicalise abbreviations
(`street→st`, `svcs→services`, `north→n`), then derive a *core name* with legal suffixes removed
(`inc, llc, pvt ltd, corp, the …`). Parse address atoms: house number, postal code (last 5–6 digit
token, which covers US ZIP and Indian PIN), unit/suite number and the first street token. All of
this happens **once per record**, so pair-level work is just array gathers.

**Generators.** Each generator covers a different failure mode. They are unioned, and each
contributes a flag that the model later uses as a feature.

| # | Generator | Key | Catches | Selectivity control |
|---|-----------|-----|---------|---------------------|
| 1 | TF-IDF char 2–4-gram kNN on core name | cosine top-25 | typos, abbreviations, word order, glued tokens | top-k, `max_df` prunes common n-grams |
| 2 | TF-IDF char kNN on name+address | cosine top-25 | generic names that only the address separates | top-k |
| 3 | Rare-name-token inverted index | exact token | heavy name noise with a surviving rare token | skip tokens in >100 S1 rows (IDF cap) |
| 4 | Phonetic compound keys | `metaphone(first)×postal`, `metaphone(first)×house`, `soundex(first)×postal` | spelling variants within the same locality | skip buckets >200 |
| 5 | Address compound keys | `house×street`, `postal×first token` | badly mangled names at the same address | skip buckets >200 |
| 6 | MinHash LSH (datasketch) | Jaccard over name+address token set | long-tail overlap patterns | LSH threshold 0.35, 64 perms |

**Budget.** The union is re-ranked with a cheap score (`0.55·cos_name + 0.45·cos_full +
0.02·#blockers`) and capped at K per record (default 40). The cap is what keeps the pipeline
linear in the number of records. On the synthetic benchmark the recall curve by cap looks like
this:

| cap K | 1 | 3 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| pair recall | 0.965 | 0.982 | 0.984 | 0.986 | 0.989 | 0.994 |

Always measure blocking with **two** numbers, both printed by the code: *pair recall*, and the
**F0.5 ceiling**, which is the macro F0.5 of an oracle that keeps exactly the true pairs among the
candidates. The ceiling is what bounds the leaderboard. Pair recall alone over-weights entities
with many matches.

**Scaling beyond ~1M records.**
- Swap the scipy matmul in `sparse_topk` for `sparse_dot_topn`, a C++ top-n sparse product.
- Or reduce the char-TF-IDF with TruncatedSVD to about 256 dimensions and run FAISS IVF/HNSW.
- Shard by postal prefix when postal coverage is high.

All generators already run in chunks, and the key-based ones are equi-joins.

### 1.2 Pair scoring: Stage 2A (GBM) vs Stage 2B (cross-encoder)

**Stage 2A: feature-engineered LightGBM** (`src/features.py`, `src/train.py`). About 107
features, all vectorised:

- **Lexical name:** Jaro-Winkler, Levenshtein ratio, partial ratio, token sort/set ratio (rapidfuzz
  `cpdist`, multi-threaded); compact-string JW (catches `JACKSONFCAFE`); first-token JW; full-name
  ratio including legal suffix.
- **Vector:** char-TF-IDF cosine on name, address and full string; word-TF-IDF cosine on name and
  address.
- **IDF-weighted set overlap** (sparse incidence matrices): weighted Jaccard, containment in each
  direction, and the *absolute unexplained IDF mass* on each side. That last one is the feature
  that catches `ValueMart` vs `ValueMart Annex` at the same address.
- **Address atoms:** house / postal / unit / street equality (NaN when missing, so the model can
  tell "missing" apart from "different"); numeric-token common/only counts for address *and* name
  (store numbers such as `#0452`).
- **Phonetic:** metaphone / soundex of the first token, sorted-metaphone string ratio.
- **Acronym flag** (`IBM` ↔ `International Business Machines`).
- **Chain-ness:** log frequency of the core name among S1 rows. A common name means only the
  address may decide.
- **Blocking provenance:** which generators fired, kNN ranks, cheap score and rank.
- **Context (the most important group):** rank of the pair among the record's candidates and among
  the S1 entity's candidates, and the *margin to the best competing candidate*. This lets the model
  learn exclusivity softly. `combo_gap_rec` is the #1 feature by gain.

**Stage 2B: cross-encoder (DeBERTa-v3-small / MiniLM-L12)** scoring
`"[name] | [address]" [SEP] "[name] | [address]"`.

| | 2A GBM | 2B Cross-encoder |
|---|---|---|
| Strength | exact atoms (house no., postal, store no.), rarity, context, calibration | semantic variants (`Doctor`↔`Clinic`, transliterations, reordered fragments) |
| Weakness | needs hand features for every noise type | weak at digit equality; no candidate context; uncalibrated |
| Throughput (CPU) | ~10⁶ pairs in seconds | ~10²–10³ pairs/s. GPU and top-k pruning are mandatory |
| Data need | a few thousand positives are enough | benefits from ≥10⁴ pairs plus hard negatives |
| Macro-F0.5 fit | probabilities feed straight into expected-F decisions | needs Platt/isotonic calibration |

**Recommendation: use 2A as the backbone and 2B as a stacked feature, not a replacement.** How to
wire in 2B:

1. Train the cross-encoder on the *same entity folds*, using blocking candidates as hard negatives
   (the top 5 non-matching candidates per record) and symmetric augmentation (swap sides).
   Suggested settings: 2–3 epochs, lr 2e-5, max_len 96.
2. Produce OOF logits for the top ~5 candidates per record. That is 5 × (records) pairs, which is
   cheap.
3. Add `ce_logit`, `ce_rank_rec` and `ce_gap_rec` as columns and retrain the GBM.

Expect gains concentrated on the name-only / address-missing records, which are the dominant
false-negative class in our error analysis.

*Compliance note:* pretrained weights are not "external data lookups", but confirm with the
organisers before using them. The pipeline scores well without them.

---

## 2. Singletons and threshold tuning for macro F0.5

### 2.1 Why the F1 recipe (threshold 0.5) is wrong here

For F_β the optimal plug-in threshold satisfies `t* = F*/(1+β²)`. This is the F_β
generalisation of the classic `F1*/2` result (Lipton et al., 2014). Adding one candidate with
match probability `p` to an entity's current prediction (TP true positives, size k) raises the
expected F0.5 only if

```
1.25(TP+p)/(0.25|T|+k+1)  >  1.25·TP/(0.25|T|+k)    ⇔    p  >  TP/(0.25|T|+k)  =  F_current/1.25
```

So once an entity is already well served (F≈0.95), extra candidates need p > 0.76. For an entity
with a **single lone candidate**, predicting it earns `p` (the chance it is the true match) and
predicting nothing earns `1−p` (the chance it is a singleton). The break-even there is 0.5. **No
single global threshold is optimal for both situations.** That is why we tune a per-entity rule.

### 2.2 Decision layer (`src/postprocess.py`)

1. **Exclusivity:** each record is only allowed to go to its arg-max S1 entity. The search treats
   this as optional, and it rarely hurts.
2. **Expected-F0.5 subset selection per entity.** Sort candidates by p and compute
   `E[F|k] ≈ 1.25·Σ_{i≤k} p_i / (0.25·Σ_i p_i + k)` for k ≥ 1, and `E[F|0] = Π(1−p_i)`, which is
   the explicit **singleton probability**. Predict the top-k* with k* = argmax. A probability floor
   `t` is tuned on top of this.
3. **Search:** grid over `{threshold, expected_f} × {exclusive on/off} × t ∈ [0.02, 0.98]`. The
   objective counts **all** S1 entities, including the ones whose true matches blocking never
   proposed, so recall loss is priced in. The threshold returned is the median of the plateau
   within 5e-4 of the best, which is more robust to calibration drift on test.
4. **Calibration matters:** expected-F assumes calibrated p. Folds are averaged from
   `binary_logloss` models, which calibrate well (OOF logloss 0.004). If a non-logloss model is
   ever swapped in, add isotonic calibration on OOF.

**Singleton safety net.** Beyond `E[F|0]`, the context features (margin to the best competitor,
name frequency) teach the model to down-weight "best of a bad bunch" candidates. Two possible
extensions:
- an entity-level singleton classifier (features: max p, gap between the top two, n candidates,
  name frequency) that vetoes weak entities;
- **graph consistency**, which only accepts an S3 record whose best S2 neighbour is assigned to the
  same S1 entity.

---

## 3. Validation

- **Entity-grouped, stratified K-fold** (`make_entity_folds`). Folds are assigned per **S1
  entity**, stratified on its match count (0 / 1 / 2 / 3+), so every fold has the same singleton
  rate. All candidate pairs of an entity inherit its fold, so a match set is never split across
  train and validation.
- **No label leakage through features.** Blocking and features are unsupervised (TF-IDF is fitted
  on the split's own text, with no labels), and the context features use similarities, not labels.
- **Nested decision tuning.** The rule and threshold are tuned on the other folds' OOF and scored on
  the held-out fold. This is the honest CV number (`nested_cv_macro_f05`). The "tuned on all OOF"
  number is slightly optimistic.
- **Sanity:** the synthetic hold-out (0.968) matched nested CV (0.966–0.970). On the real data,
  confirm that CV tracks the leaderboard before trusting small deltas; ±0.003 is noise at this
  scale. Also run adversarial validation (train vs test features) to detect vendor drift.

---

## 4. Prioritised next steps once the real data lands

1. **Look at the data.** Check column semantics, singleton rate, matches per entity, S2 vs S3 noise
   profiles, and address format (India vs US). Extend `ABBREV` / `LEGAL_SUFFIXES` accordingly; this
   is the highest-ROI work.
2. **Blocking audit.** Read through the missed true pairs (`blocking_report`) and add a generator
   for each miss pattern.
3. **Error analysis on OOF** (`oof_predictions.tsv`). Bucket FPs/FNs by pattern and add one feature
   per pattern.
4. **2-stage stacking.** Recompute the context features on stage-1 OOF probabilities (rank/gap of
   p within record and entity) and train a stage-2 GBM.
5. **Cross-encoder stacked feature** (§1.2).
6. **Seed / fold bagging, plus a CatBoost blend.** Average the probabilities *before* the decision
   layer and retune the rule.
