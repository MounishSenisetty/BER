"""Gradient-boosting backends behind one interface.

backend = "auto"     -> XGBoost on the GPU when XGBoost has CUDA and a GPU is visible, else LightGBM
          "xgboost"  -> XGBoost (GPU if available, else CPU hist)
          "lightgbm" -> LightGBM on CPU

Both are permissively licensed (XGBoost Apache-2.0, LightGBM MIT). A trained model is stored as
(backend, serialized booster, best iteration) so inference can load it without the training config.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import LOG


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return out.returncode == 0 and "GPU" in out.stdout
    except Exception:
        return False


def _xgb_cuda() -> bool:
    try:
        import xgboost as xgb
        return bool(xgb.build_info().get("USE_CUDA")) and gpu_available()
    except Exception:
        return False


def resolve_backend(backend: str) -> str:
    if backend == "auto":
        backend = "xgboost" if _xgb_cuda() else "lightgbm"
    LOG.info("Model backend: %s%s", backend, " (GPU)" if backend == "xgboost" and _xgb_cuda() else "")
    return backend


def _xgb_params(cfg) -> Dict:
    p = cfg.params
    return {
        "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
        "device": "cuda" if _xgb_cuda() else "cpu", "eta": p.get("learning_rate", 0.05),
        "grow_policy": "lossguide", "max_leaves": p.get("num_leaves", 127), "max_depth": 0,
        "min_child_weight": 5, "subsample": p.get("bagging_fraction", 0.8),
        "colsample_bytree": p.get("feature_fraction", 0.8), "lambda": p.get("lambda_l2", 1.0),
        "max_bin": p.get("max_bin", 127), "seed": cfg.seed, "nthread": os.cpu_count() or 1,
    }


def train_fold(backend: str, cfg, Xtr: pd.DataFrame, ytr, Xva: pd.DataFrame, yva) -> Tuple[Tuple, np.ndarray, pd.DataFrame]:
    """Returns (serialized model, validation predictions, importance frame)."""
    if backend == "xgboost":
        import xgboost as xgb
        dtr = xgb.QuantileDMatrix(Xtr, ytr, max_bin=_xgb_params(cfg)["max_bin"])
        dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr, max_bin=_xgb_params(cfg)["max_bin"])
        b = xgb.train(_xgb_params(cfg), dtr, cfg.num_boost_round, evals=[(dva, "valid")],
                      early_stopping_rounds=cfg.early_stopping_rounds, verbose_eval=False)
        best = int(b.best_iteration) + 1
        pred = b.predict(dva, iteration_range=(0, best))
        gain, weight = b.get_score(importance_type="total_gain"), b.get_score(importance_type="weight")
        imp = pd.DataFrame({"feature": Xtr.columns, "gain": [gain.get(f, 0.0) for f in Xtr.columns],
                            "split": [weight.get(f, 0.0) for f in Xtr.columns]})
        return ("xgboost", bytes(b.save_raw("json")), best), pred, imp
    import lightgbm as lgb
    params = dict(cfg.params, seed=cfg.seed, num_threads=os.cpu_count() or 1)
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    b = lgb.train(params, dtr, cfg.num_boost_round, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False), lgb.log_evaluation(0)])
    pred = b.predict(Xva, num_iteration=b.best_iteration)
    imp = pd.DataFrame({"feature": Xtr.columns, "gain": b.feature_importance("gain"),
                        "split": b.feature_importance("split")})
    return ("lightgbm", b.model_to_string(num_iteration=b.best_iteration), b.best_iteration), pred, imp


class Ensemble:
    """Average of the fold models; loads either backend."""

    def __init__(self, models: List):
        self.parts = []
        for m in models:
            if isinstance(m, str):                    # bundles from before the backend switch
                m = ("lightgbm", m, None)
            kind, blob, best = m
            if kind == "xgboost":
                import xgboost as xgb
                b = xgb.Booster()
                b.load_model(bytearray(blob))
                b.set_param({"device": "cuda" if _xgb_cuda() else "cpu", "nthread": os.cpu_count() or 1})
                self.parts.append((kind, b, best))
            else:
                import lightgbm as lgb
                self.parts.append((kind, lgb.Booster(model_str=blob), best))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = np.zeros(len(X), dtype=np.float64)
        dm = None
        for kind, b, best in self.parts:
            if kind == "xgboost":
                import xgboost as xgb
                dm = dm if dm is not None else xgb.DMatrix(X)
                p += b.predict(dm, iteration_range=(0, best))
            else:
                p += b.predict(X)
        return p / max(len(self.parts), 1)
