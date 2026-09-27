"""
Model training utilities for the entity matching pipeline.
Handles entity-level train/val splitting and CatBoost model training.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Sequence
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from catboost import CatBoostClassifier, Pool

from business_entity_resolution.matching.features import FEATURE_NAMES


def entity_level_split(
    s1_ids: Union[Sequence[str], np.ndarray],
    val_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """
    Performs entity-level train/val split on unique S1 entity IDs.
    Ensures all candidate pairs for any S1 entity fall strictly within either train or val.

    Parameters
    ----------
    s1_ids : sequence or array-like
        Array or list of S1 entity IDs (may contain duplicates).
    val_fraction : float, default 0.20
        Fraction of unique S1 entities to allocate to validation.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[Set[str], Set[str]]
        (train_s1_ids, val_s1_ids) sets.
    """
    unique_s1 = np.unique(s1_ids).tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_s1)
    split_idx = int(len(unique_s1) * (1.0 - val_fraction))
    train_ids = set(unique_s1[:split_idx])
    val_ids = set(unique_s1[split_idx:])
    return train_ids, val_ids


def create_catboost_matcher(
    iterations: int = 1500,
    learning_rate: float = 0.05,
    depth: int = 8,
    l2_leaf_reg: float = 3.0,
    thread_count: int = 8,
    random_seed: int = 42,
    auto_class_weights: Optional[str] = "Balanced",
    verbose: int = 100,
) -> CatBoostClassifier:
    """
    Instantiates a CatBoostClassifier configured for binary pairwise entity matching.
    """
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights=auto_class_weights,
        task_type="CPU",
        thread_count=thread_count,
        verbose=verbose,
        early_stopping_rounds=100,
        random_seed=random_seed,
    )


def train_catboost_matcher(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str] = FEATURE_NAMES,
    model_save_path: Optional[Path] = None,
    **catboost_kwargs,
) -> Tuple[CatBoostClassifier, Dict[str, float]]:
    """
    Trains CatBoost on train features and evaluates against validation features.

    Returns
    -------
    Tuple[CatBoostClassifier, Dict[str, float]]
        Trained model and dictionary of evaluation metrics.
    """
    model = create_catboost_matcher(**catboost_kwargs)
    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    val_pool = Pool(X_val, y_val, feature_names=feature_names)

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    if model_save_path is not None:
        model_save_path = Path(model_save_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_save_path))

    metrics = {
        "best_iteration": int(model.get_best_iteration()),
        "best_score": model.get_best_score(),
    }
    return model, metrics


def get_feature_importances(
    model: CatBoostClassifier,
    feature_names: List[str] = FEATURE_NAMES,
) -> pd.DataFrame:
    """
    Returns a DataFrame of feature importances sorted descending.
    """
    importances = model.get_feature_importance()
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    return df


try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def create_lightgbm_matcher(
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    max_depth: int = 8,
    min_child_samples: int = 50,
    subsample: float = 0.8,
    subsample_freq: int = 1,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 3.0,
    thread_count: int = 8,
    random_seed: int = 42,
    verbose: int = -1,
) -> Dict[str, Any]:
    """
    Returns a dictionary of LightGBM hyperparameters for binary pairwise entity matching.
    """
    return {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "subsample_freq": subsample_freq,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "n_jobs": thread_count,
        "random_state": random_seed,
        "verbose": verbose,
    }


def train_lightgbm_matcher(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str] = FEATURE_NAMES,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 100,
    model_save_path: Optional[Path] = None,
    verbose_eval: int = 200,
    **lgb_kwargs,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains a LightGBM Booster on train features and evaluates against validation features.

    Returns
    -------
    Tuple[lgb.Booster, Dict[str, Any]]
        Trained Booster model and dictionary of evaluation metrics.
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Run 'pip install lightgbm'.")

    params = create_lightgbm_matcher(**lgb_kwargs)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_names)

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
    ]
    if verbose_eval > 0:
        callbacks.append(lgb.log_evaluation(period=verbose_eval))

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    if model_save_path is not None:
        model_save_path = Path(model_save_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(model_save_path))

    best_iter = booster.best_iteration if booster.best_iteration is not None else num_boost_round
    best_score = booster.best_score if hasattr(booster, "best_score") else {}
    metrics = {
        "best_iteration": int(best_iter),
        "best_score": best_score,
    }
    return booster, metrics


def get_lightgbm_feature_importances(
    booster: Any,
    feature_names: List[str] = FEATURE_NAMES,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Returns a DataFrame of LightGBM feature importances sorted descending.
    """
    importances = booster.feature_importance(importance_type=importance_type)
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    return df
