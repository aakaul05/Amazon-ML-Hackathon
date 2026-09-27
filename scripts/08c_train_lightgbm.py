"""
scripts/08c_train_lightgbm.py
=============================
Task 8C: 5-Fold Entity-Level LightGBM Training and Threshold Optimization.

1. Samples training data from precomputed Task 8B feature parquets:
   - 100% positives
   - 3.0x hard negatives (name similarity >= 0.50 or JW >= 0.70 or passes >= 2)
   - 1.0x easy negatives
2. Performs 5-fold entity-level cross-validation on unique S1 entities (zero S1 leakage).
3. Trains 5 LightGBM Booster models with early stopping.
4. Generates OOF predictions on the training pool.
5. Optimizes decision threshold strictly for Macro F0.5.
6. Saves:
   - 5 LightGBM models: data/student_resource/outputs/matching/models/lightgbm/lightgbm_matcher_fold1..5.txt
   - metrics_lightgbm.json
   - feature_importance_lightgbm.csv
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.matching import (
    FEATURE_NAMES,
    compute_entity_level_metrics,
    create_lightgbm_matcher,
    train_lightgbm_matcher,
    get_lightgbm_feature_importances,
)

OUTPUT_BASE = REPO_ROOT / "data" / "student_resource" / "outputs" / "matching"
FEATURES_DIR = OUTPUT_BASE / "features"
MODELS_DIR = OUTPUT_BASE / "models" / "lightgbm"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS = 5
HARD_NEG_RATIO = 3.0
EASY_NEG_RATIO = 1.0

LGBM_LEARNING_RATE = 0.05
LGBM_NUM_LEAVES = 63
LGBM_MAX_DEPTH = 8
LGBM_NUM_BOOST_ROUND = 1500
LGBM_EARLY_STOPPING = 100
LGBM_THREADS = 8

COARSE_THRESHOLDS = np.round(np.arange(0.50, 1.00, 0.05), 2)
FINE_STEP = 0.01
FINE_RADIUS = 0.05


def collect_training_samples(
    source_feature_dirs: List[Path],
    hard_neg_ratio: float = HARD_NEG_RATIO,
    easy_neg_ratio: float = EASY_NEG_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collects balanced training data from existing Task 8B feature parquets."""
    print("=" * 75)
    print("COLLECTING BALANCED TRAINING DATA FROM EXISTING FEATURE PARQUETS")
    print("=" * 75)

    pos_dfs = []
    hard_neg_dfs = []
    easy_neg_dfs = []
    rng = np.random.RandomState(random_seed)

    for src_dir in source_feature_dirs:
        if not src_dir.is_dir():
            print(f"Warning: Feature directory not found: {src_dir}")
            continue

        parquet_files = sorted(list(src_dir.glob("part_*.parquet")))
        print(f"Reading {len(parquet_files)} feature chunks from {src_dir.name}...")

        for p_file in parquet_files:
            df_chunk = pd.read_parquet(p_file)

            df_pos = df_chunk[df_chunk["label"] == 1]
            if len(df_pos) > 0:
                pos_dfs.append(df_pos)

            n_pos = len(df_pos)
            if n_pos == 0:
                continue

            df_neg = df_chunk[df_chunk["label"] == 0]
            hard_mask = (
                (df_neg["name_token_sort_sim"] >= 0.50)
                | (df_neg["name_jaro_winkler"] >= 0.70)
                | (df_neg["num_blocking_passes"] >= 2)
            )

            df_hard = df_neg[hard_mask]
            df_easy = df_neg[~hard_mask]

            n_hard_sample = min(len(df_hard), int(n_pos * hard_neg_ratio))
            if n_hard_sample > 0:
                hard_neg_dfs.append(df_hard.sample(n=n_hard_sample, random_state=rng.randint(0, 1000000)))

            n_easy_sample = min(len(df_easy), int(n_pos * easy_neg_ratio))
            if n_easy_sample > 0:
                easy_neg_dfs.append(df_easy.sample(n=n_easy_sample, random_state=rng.randint(0, 1000000)))

    if not pos_dfs:
        raise RuntimeError("No positive training pairs found in feature directories!")

    print("\nConcatenating sampled dataframes...")
    df_all_pos = pd.concat(pos_dfs, ignore_index=True)
    df_all_hard = pd.concat(hard_neg_dfs, ignore_index=True)
    df_all_easy = pd.concat(easy_neg_dfs, ignore_index=True)

    print(f"Total Positives: {len(df_all_pos):,}")
    print(f"Total Hard Negs: {len(df_all_hard):,}")
    print(f"Total Easy Negs: {len(df_all_easy):,}")

    df_train_full = pd.concat([df_all_pos, df_all_hard, df_all_easy], ignore_index=True)
    df_train_full = df_train_full.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    s1_ids = df_train_full["s1_entity_id"].values
    candidate_ids = df_train_full["candidate_entity_id"].values
    labels = df_train_full["label"].values.astype(np.int8)
    X = df_train_full[FEATURE_NAMES].values.astype(np.float32)

    print(f"\nFinal sampled training pool: {len(df_train_full):,} pairs ({np.sum(labels):,} positive)")
    return s1_ids, candidate_ids, X, labels


def create_entity_fold_mapping(unique_s1: np.ndarray, n_folds: int = N_FOLDS) -> Dict[str, int]:
    """Ensures zero S1 entity leakage across folds."""
    print("\nCreating entity-level fold mapping (KFold)...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    entity_to_fold = {}
    for fold, (_, val_idx) in enumerate(kf.split(unique_s1), 1):
        val_entities = unique_s1[val_idx]
        for eid in val_entities:
            entity_to_fold[eid] = fold
        print(f"Fold {fold}: {len(val_entities):,} validation S1 entities")
    print(f"Mapped {len(entity_to_fold):,} unique S1 entities to folds.")
    return entity_to_fold


def train_five_lightgbm_models(
    s1_ids: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    entity_to_fold: Dict[str, int],
) -> Tuple[List[lgb.Booster], np.ndarray]:
    """Trains 5-fold LightGBM ensemble and collects OOF probabilities."""
    print("\n" + "=" * 75)
    print("5-FOLD ENTITY-LEVEL CROSS-VALIDATION & LIGHTGBM TRAINING")
    print("=" * 75)

    oof_probs = np.zeros(len(s1_ids), dtype=np.float32)
    models = []
    feature_importances = []

    for fold in range(1, N_FOLDS + 1):
        t0 = time.time()
        val_mask = np.fromiter((entity_to_fold[eid] == fold for eid in s1_ids), dtype=bool, count=len(s1_ids))
        train_mask = ~val_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        print(f"\n--- Fold {fold}/{N_FOLDS} ---")
        print(f"Train: {len(X_train):,} pairs (Positives: {int(np.sum(y_train)):,})")
        print(f"Val  : {len(X_val):,} pairs (Positives: {int(np.sum(y_val)):,})")

        model_save_path = MODELS_DIR / f"lightgbm_matcher_fold{fold}.txt"

        booster, metrics = train_lightgbm_matcher(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=FEATURE_NAMES,
            num_boost_round=LGBM_NUM_BOOST_ROUND,
            early_stopping_rounds=LGBM_EARLY_STOPPING,
            learning_rate=LGBM_LEARNING_RATE,
            num_leaves=LGBM_NUM_LEAVES,
            max_depth=LGBM_MAX_DEPTH,
            thread_count=LGBM_THREADS,
            random_seed=RANDOM_SEED + fold,
            model_save_path=model_save_path,
            verbose_eval=200,
        )

        val_probs = booster.predict(X_val).astype(np.float32)
        oof_probs[val_mask] = val_probs
        models.append(booster)

        fi_df = get_lightgbm_feature_importances(booster, FEATURE_NAMES)
        feature_importances.append(fi_df)

        elapsed = (time.time() - t0) / 60
        print(f"Fold {fold} finished in {elapsed:.2f} mins (best iter: {metrics['best_iteration']})")
        del X_train, y_train, X_val, y_val, val_probs
        gc.collect()

    # Aggregate feature importances
    mean_fi = (
        pd.concat(feature_importances)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_path = MODELS_DIR / "feature_importance_lightgbm.csv"
    mean_fi.to_csv(fi_path, index=False)
    print(f"\nSaved feature importances to {fi_path}")

    return models, oof_probs


def sweep_oof_thresholds(
    s1_ids: np.ndarray,
    candidate_ids: np.ndarray,
    oof_probs: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, Dict[str, float], List[dict]]:
    """Sweeps thresholds on OOF predictions to optimize Macro F0.5."""
    print("\n" + "=" * 75)
    print("OOF THRESHOLD OPTIMIZATION (MACRO F0.5)")
    print("=" * 75)

    # Build GT dict for sampled pairs
    all_s1 = sorted(list(set(s1_ids)))
    gt_dict = {eid: set() for eid in all_s1}
    for s1, ot, y in zip(s1_ids, candidate_ids, labels):
        if y == 1:
            gt_dict[s1].add(ot)

    threshold_grid = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    records = []
    best_f05 = -1.0
    best_th = 0.95
    best_metrics = {}

    print(f"{'THRESHOLD':<12} {'PRECISION':>12} {'RECALL':>12} {'MACRO F0.5':>14} {'MACRO F1':>12} {'MATCHES':>12}")
    print("-" * 80)

    for th in threshold_grid:
        pass_mask = oof_probs >= th
        pred_dict = {eid: set() for eid in all_s1}
        for s1, ot in zip(s1_ids[pass_mask], candidate_ids[pass_mask]):
            pred_dict[s1].add(ot)

        m = compute_entity_level_metrics(pred_dict, gt_dict, all_s1_ids=set(all_s1))
        m["threshold"] = th
        m["predicted_matches"] = int(pass_mask.sum())
        records.append(m)

        is_best = ""
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_th = th
            best_metrics = m
            is_best = " <-- BEST"

        print(f"{th:<12.2f} {m['macro_precision']:>12.4f} {m['macro_recall']:>12.4f} {m['macro_f05']:>14.4f} {m['macro_f1']:>12.4f} {m['predicted_matches']:>12,}{is_best}")

    print("-" * 80)
    print(f"Optimal LightGBM Threshold: {best_th:.2f} (Macro F0.5: {best_f05:.4f})")
    return best_th, best_metrics, records


def main():
    print("=" * 80)
    print("TASK 8C: LIGHTGBM 5-FOLD ENSEMBLE TRAINING & THRESHOLD OPTIMIZATION")
    print("=" * 80)

    feature_dirs = [FEATURES_DIR / "s1_s2", FEATURES_DIR / "s1_s3"]
    t_start = time.time()

    # 1. Collect training samples from 8B feature parquets
    s1_ids, candidate_ids, X, y = collect_training_samples(feature_dirs)

    # 2. Entity-level split
    unique_s1 = np.unique(s1_ids)
    entity_to_fold = create_entity_fold_mapping(unique_s1, n_folds=N_FOLDS)

    # 3. Train 5 LightGBM models
    models, oof_probs = train_five_lightgbm_models(s1_ids, X, y, entity_to_fold)

    # 4. Sweep thresholds on OOF
    best_th, best_metrics, threshold_records = sweep_oof_thresholds(s1_ids, candidate_ids, oof_probs, y)

    # 5. Save comprehensive metrics
    total_elapsed = (time.time() - t_start) / 60
    metrics_summary = {
        "model_type": "lightgbm",
        "n_folds": N_FOLDS,
        "features": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "total_training_pairs": len(s1_ids),
        "total_positives": int(np.sum(y)),
        "hard_neg_ratio": HARD_NEG_RATIO,
        "easy_neg_ratio": EASY_NEG_RATIO,
        "hyperparameters": {
            "learning_rate": LGBM_LEARNING_RATE,
            "num_leaves": LGBM_NUM_LEAVES,
            "max_depth": LGBM_MAX_DEPTH,
            "num_boost_round": LGBM_NUM_BOOST_ROUND,
            "early_stopping_rounds": LGBM_EARLY_STOPPING,
        },
        "total_training_time_minutes": round(total_elapsed, 2),
        "optimal_threshold": best_th,
        "optimal_oof_metrics": best_metrics,
        "all_threshold_records": threshold_records,
    }

    metrics_path = MODELS_DIR / "metrics_lightgbm.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nSaved LightGBM metrics and threshold configuration to: {metrics_path}")
    print("=" * 80)
    print(f"LIGHTGBM TRAINING COMPLETED IN {total_elapsed:.1f} MINUTES.")
    print("=" * 80)


if __name__ == "__main__":
    main()
