"""
Model training utilities for the entity matching pipeline.
Handles entity-level train/val splitting and CatBoost model training.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Sequence
import numpy as np
import pandas as pd
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
    unique_s1 = list(set(s1_ids))
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
