"""
Script 08c: CatBoost Model Training, Threshold Optimization & Full Inference (Task 8C).
1. Samples balanced training set (all positives + stratified hard & easy negatives).
2. Performs entity-level train/val split (80/20).
3. Trains CatBoost binary classifier.
4. Sweeps decision thresholds to maximize Macro F0.5.
5. Performs streaming inference across all 264M candidate pairs.
6. Computes final Macro F0.5 on the full dataset and saves predictions.
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
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.matching import (
    FEATURE_NAMES,
    load_ground_truth_dict,
    entity_level_split,
    train_catboost_matcher,
    get_feature_importances,
    sweep_thresholds,
    compute_entity_level_metrics,
)


def find_dataset_dir() -> Path:
    p = os.environ.get("BER_DATA_DIR")
    if p and Path(p).is_dir():
        return Path(p)
    candidates = [
        REPO_ROOT / "dataset" / "student_resource" / "dataset" / "train",
        REPO_ROOT / "data" / "student_resource" / "dataset" / "train",
        Path.home() / "Amazon-ML-Hackathon" / "dataset" / "student_resource" / "dataset" / "train",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return REPO_ROOT / "dataset" / "student_resource" / "dataset" / "train"


OUTPUT_BASE = REPO_ROOT / "data" / "student_resource" / "outputs" / "matching"
FEATURES_DIR = OUTPUT_BASE / "features"
MODELS_DIR = OUTPUT_BASE / "models"
PREDS_DIR = OUTPUT_BASE / "predictions"
DIAG_DIR = OUTPUT_BASE / "diagnostics"

for d in [MODELS_DIR, PREDS_DIR, DIAG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def collect_training_samples(
    source_feature_dirs: List[Path],
    hard_neg_ratio: float = 3.0,
    easy_neg_ratio: float = 1.0,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scans feature chunk parquets and collects:
    - 100% of positive pairs
    - Hard negatives (sampled based on similarity signals)
    - Easy negatives (random sample)
    """
    print("=" * 75)
    print("COLLECTING BALANCED TRAINING DATA FROM FEATURE PARQUETS")
    print("=" * 75)

    pos_dfs = []
    hard_neg_dfs = []
    easy_neg_dfs = []

    rng = np.random.RandomState(random_seed)

    for src_dir in source_feature_dirs:
        parquet_files = sorted(list(src_dir.glob("part_*.parquet")))
        print(f"Reading {len(parquet_files)} chunks from {src_dir.name}...")

        for p_file in parquet_files:
            df_chunk = pd.read_parquet(p_file)

            # 1. All positives
            df_pos = df_chunk[df_chunk["label"] == 1]
            if len(df_pos) > 0:
                pos_dfs.append(df_pos)

            n_pos = len(df_pos)
            if n_pos == 0:
                continue

            # 2. Hard negatives: high similarity or multi-pass but label == 0
            df_neg = df_chunk[df_chunk["label"] == 0]
            hard_mask = (df_neg["name_token_sort_sim"] >= 0.50) | (df_neg["name_jaro_winkler"] >= 0.70) | (df_neg["num_blocking_passes"] >= 2)
            df_hard = df_neg[hard_mask]
            df_easy = df_neg[~hard_mask]

            n_hard_sample = min(len(df_hard), int(n_pos * hard_neg_ratio))
            if n_hard_sample > 0:
                hard_neg_dfs.append(df_hard.sample(n=n_hard_sample, random_state=rng.randint(0, 1000000)))

            n_easy_sample = min(len(df_easy), int(n_pos * easy_neg_ratio))
            if n_easy_sample > 0:
                easy_neg_dfs.append(df_easy.sample(n=n_easy_sample, random_state=rng.randint(0, 1000000)))

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
    cand_ids = df_train_full["candidate_entity_id"].values
    labels = df_train_full["label"].values.astype(np.int8)
    X = df_train_full[FEATURE_NAMES].values.astype(np.float32)

    print(f"Final sampled training pool: {len(df_train_full):,} pairs ({np.sum(labels):,} positive)")
    return s1_ids, cand_ids, X, labels


def main():
    train_dir = find_dataset_dir()
    print(f"Train Dataset Dir: {train_dir}")
    print(f"Features Base Dir: {FEATURES_DIR}")

    s2_features_dir = FEATURES_DIR / "s1_s2"
    s3_features_dir = FEATURES_DIR / "s1_s3"

    if not s2_features_dir.exists() and not s3_features_dir.exists():
        print(f"ERROR: Feature directories not found in {FEATURES_DIR}. Run 08b_full_features.py first.")
        return

    # Load ground truth for evaluation
    gt_file = train_dir / "train_ground_truth.tsv"
    if not gt_file.is_file():
        alt_gt = REPO_ROOT / "dataset" / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"
        if alt_gt.is_file():
            gt_file = alt_gt

    print(f"Loading ground truth dictionary from {gt_file}...")
    gt_dict = load_ground_truth_dict(gt_file)

    # 1. Collect training samples
    feature_dirs = [d for d in [s2_features_dir, s3_features_dir] if d.exists()]
    s1_ids, cand_ids, X, y = collect_training_samples(feature_dirs)

    # 2. Entity-level train/val split
    print("\nSplitting unique S1 entities into Train (80%) and Val (20%)...")
    train_s1, val_s1 = entity_level_split(s1_ids, val_fraction=0.20, seed=42)
    train_mask = np.fromiter((eid in train_s1 for eid in s1_ids), dtype=bool, count=len(s1_ids))
    val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Train set: {len(X_train):,} pairs (Positives: {int(np.sum(y_train)):,})")
    print(f"Val set  : {len(X_val):,} pairs (Positives: {int(np.sum(y_val)):,})")

    # 3. Train CatBoost Model
    print("\n" + "=" * 75)
    print("TRAINING CATBOOST MATCHER")
    print("=" * 75)
    t0 = time.time()
    model_save_path = MODELS_DIR / "catboost_matcher_v1.cbm"
    model, metrics = train_catboost_matcher(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        iterations=2000,
        learning_rate=0.06,
        depth=8,
        l2_leaf_reg=3.0,
        thread_count=8,
        model_save_path=model_save_path,
        verbose=100,
    )
    print(f"Training completed in {(time.time() - t0)/60:.2f} mins.")

    # Save feature importances
    df_fi = get_feature_importances(model, FEATURE_NAMES)
    print("\nFeature Importances:")
    print(df_fi.to_string(index=False))
    df_fi.to_csv(DIAG_DIR / "feature_importance_v1.csv", index=False)

    # 4. Sweep Thresholds on Validation S1 entities
    print("\n" + "=" * 75)
    print("THRESHOLD OPTIMIZATION FOR MACRO F0.5")
    print("=" * 75)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_s1_arr = s1_ids[val_mask]
    val_cand_arr = cand_ids[val_mask]

    sweep_df, best_threshold = sweep_thresholds(
        s1_ids=val_s1_arr,
        candidate_ids=val_cand_arr,
        probabilities=val_probs,
        ground_truth_by_s1=gt_dict,
        val_s1_ids=val_s1,
    )

    print("\nValidation Threshold Sweep Table:")
    print(sweep_df.to_string(index=False))
    sweep_df.to_csv(DIAG_DIR / "threshold_sweep_v1.csv", index=False)

    best_row = sweep_df.loc[sweep_df["threshold"] == best_threshold].iloc[0]
    print(f"\nOPTIMAL THRESHOLD: {best_threshold:.2f}")
    print(f"Validation Macro F0.5: {best_row['macro_f05']:.4f}")
    print(f"Validation Macro P   : {best_row['macro_precision']:.4f}")
    print(f"Validation Macro R   : {best_row['macro_recall']:.4f}")

    # Free memory before inference
    del X, y, X_train, y_train, X_val, y_val, val_probs
    gc.collect()

    # 5. Streaming Full Inference Across All Features
    print("\n" + "=" * 75)
    print(f"FULL INFERENCE ACROSS ALL CANDIDATES (Threshold = {best_threshold:.2f})")
    print("=" * 75)

    all_predictions_by_s1 = defaultdict(set)

    for src_name, src_dir in [("s2", s2_features_dir), ("s3", s3_features_dir)]:
        if not src_dir.exists():
            continue

        p_files = sorted(list(src_dir.glob("part_*.parquet")))
        print(f"\nScoring {len(p_files)} feature chunks for {src_name.upper()}...")
        t_src_start = time.time()
        matched_pairs_s1 = []
        matched_pairs_cand = []
        matched_probs = []

        for p_idx, p_file in enumerate(p_files):
            df_part = pd.read_parquet(p_file)
            X_part = df_part[FEATURE_NAMES].values.astype(np.float32)
            probs = model.predict_proba(X_part)[:, 1]

            keep_mask = probs >= best_threshold
            if np.any(keep_mask):
                k_s1 = df_part["s1_entity_id"].values[keep_mask]
                k_cand = df_part["candidate_entity_id"].values[keep_mask]
                k_pr = probs[keep_mask]

                matched_pairs_s1.extend(k_s1)
                matched_pairs_cand.extend(k_cand)
                matched_probs.extend(k_pr)

                for s, c in zip(k_s1, k_cand):
                    all_predictions_by_s1[s].add(c)

            if (p_idx + 1) % 25 == 0 or (p_idx + 1) == len(p_files):
                print(f"  [{p_idx+1:03d}/{len(p_files):03d}] Processed. Total matches kept so far: {len(matched_pairs_s1):,}")

        # Save predictions parquet for this target source
        df_pred_src = pd.DataFrame({
            "s1_entity_id": matched_pairs_s1,
            "matched_entity_id": matched_pairs_cand,
            "probability": matched_probs,
        })
        pred_out_file = PREDS_DIR / f"s1_{src_name}_matches.parquet"
        df_pred_src.to_parquet(pred_out_file, engine="pyarrow", compression="snappy", index=False)
        print(f"Saved {len(df_pred_src):,} {src_name.upper()} predictions to {pred_out_file.name} in {(time.time() - t_src_start)/60:.2f} mins")

    # 6. Overall Full Dataset Macro F0.5 Evaluation
    print("\n" + "=" * 75)
    print("FINAL FULL DATASET EVALUATION (Macro F0.5)")
    print("=" * 75)

    final_metrics = compute_entity_level_metrics(
        predictions_by_s1=all_predictions_by_s1,
        ground_truth_by_s1=gt_dict,
    )

    print(f"  Macro Precision : {final_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall    : {final_metrics['macro_recall']:.4f}")
    print(f"  Macro F0.5 Score: {final_metrics['macro_f05']:.4f}")
    print(f"  Macro F1 Score  : {final_metrics['macro_f1']:.4f}")
    print(f"  Evaluated S1 IDs: {final_metrics['num_evaluated_entities']:,}")

    metrics_output = {
        "best_threshold": float(best_threshold),
        "val_macro_f05": float(best_row["macro_f05"]),
        "full_macro_f05": float(final_metrics["macro_f05"]),
        "full_macro_precision": float(final_metrics["macro_precision"]),
        "full_macro_recall": float(final_metrics["macro_recall"]),
        "full_macro_f1": float(final_metrics["macro_f1"]),
        "evaluated_entities": int(final_metrics["num_evaluated_entities"]),
    }

    with open(MODELS_DIR / "metrics_v1.json", "w") as f:
        json.dump(metrics_output, f, indent=2)

    print(f"\nSaved metrics to {MODELS_DIR / 'metrics_v1.json'}")
    print("Task 8 Matching Pipeline Complete!")


if __name__ == "__main__":
    main()
