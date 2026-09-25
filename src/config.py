"""Central configuration. Every knob used by blocking / features / training lives here so that
train and inference are guaranteed to run the exact same pipeline.

Defaults are sized for the competition data on a Kaggle CPU notebook (4 cores, ~30 GB RAM):
~2M Source 1 entities and ~10M Source 2/3 records per split, processed country by country and
in record chunks so memory stays bounded."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class BlockingConfig:
    partition_by_country: bool = True    # only compare records with the same country label
    chunk_records: int = 300_000         # records processed per chunk (bounds peak memory)

    # --- char n-gram TF-IDF (hashed, idf fitted on Source 1 of the split) ---
    ngram_range: tuple = (3, 3)
    hash_features: int = 2 ** 20

    # --- dense kNN: TruncatedSVD of the TF-IDF + FAISS inner-product index ---
    svd_dim: int = 128
    svd_fit_rows: int = 300_000
    faiss_nprobe: int = 24
    exact_knn_below: int = 20_000        # smaller S1 partitions use exact (flat) search
    name_topk: int = 10                  # S1 neighbours per record, name space
    full_topk: int = 10                  # S1 neighbours per record, name+address space

    # --- sparse key index: rare tokens, token pairs, phonetic/address compound keys ---
    key_max_df: int = 100                # drop keys shared by more S1 rows than this
    key_topk: int = 15
    key_hash_features: int = 2 ** 23

    # --- final budget: union re-ranked by a cheap score and capped per record ---
    max_candidates_per_record: int = 10
    addr_keep: int = 2                   # + best-address candidates (trade names / acronyms) ...
    addr_keep_min_cos: float = 0.5       # ... when their address cosine is at least this
    key_keep: int = 2                    # + best key-overlap candidates


@dataclass
class ModelConfig:
    n_folds: int = 5
    seed: int = 42
    train_entities: int = 120_000        # S1 entities sampled for training (0 = all)
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "verbose": -1,
    })


@dataclass
class DecisionConfig:
    """Post-processing search space (tuned on OOF predictions for macro F0.5)."""
    modes: tuple = ("threshold", "expected_f")
    # hard exclusivity needs every candidate of a record scored; training only scores pairs of the
    # sampled entities, so it cannot be tuned honestly -- the model learns it softly via context features
    exclusive_options: tuple = (False,)
    grid: tuple = tuple(round(0.02 + 0.01 * i, 2) for i in range(97))  # 0.02 .. 0.98


@dataclass
class PipelineConfig:
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    n_jobs: int = -1

    def to_dict(self) -> dict:
        return asdict(self)
