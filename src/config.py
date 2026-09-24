"""Central configuration. Every knob used by blocking / features / training lives here
so that train and inference are guaranteed to run the exact same pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class BlockingConfig:
    # --- TF-IDF char n-gram nearest neighbours (the recall workhorse) ---
    ngram_range: tuple = (2, 4)
    tfidf_max_df: float = 0.25          # drop ultra-common n-grams -> sparser matmul, faster
    name_topk: int = 25                 # S1 neighbours per record on the name space
    full_topk: int = 25                 # S1 neighbours per record on the name+address space
    min_sim: float = 0.05
    chunk_size: int = 2000              # query rows per sparse matmul chunk

    # --- key-based blockers (inverted indexes) ---
    token_max_bucket: int = 100         # skip name tokens shared by more S1 rows than this
    key_max_bucket: int = 200           # skip phonetic / address keys with bigger S1 buckets

    # --- MinHash LSH over name+address token sets ---
    use_lsh: bool = True
    lsh_num_perm: int = 64
    lsh_threshold: float = 0.35

    # --- final budget ---
    max_candidates_per_record: int = 40  # union is re-ranked by a cheap score and capped


@dataclass
class ModelConfig:
    n_folds: int = 5
    seed: int = 42
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
    })


@dataclass
class DecisionConfig:
    """Post-processing search space (tuned on OOF predictions for macro F0.5)."""
    modes: tuple = ("threshold", "expected_f")
    exclusive_options: tuple = (True, False)
    grid: tuple = tuple(round(0.02 + 0.01 * i, 2) for i in range(97))  # 0.02 .. 0.98


@dataclass
class PipelineConfig:
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    n_jobs: int = -1

    def to_dict(self) -> dict:
        return asdict(self)
