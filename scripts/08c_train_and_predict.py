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
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.matching import (
    FEATURE_NAMES,
    load_ground_truth_dict,
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

    # 2. 5-Fold Entity-Level Split & Training
    print("\n" + "=" * 75)
    print("5-FOLD CROSS-VALIDATION & MODEL TRAINING")
    print("=" * 75)

    unique_s1_sorted = np.array(sorted(list(set(s1_ids))))
    print(f"Total sampled pool pairs: {len(s1_ids):,} across {len(unique_s1_sorted):,} unique S1 entities")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(s1_ids), dtype=np.float32)
    models_list = []
    feature_importances_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_s1_sorted), 1):
        t0 = time.time()
        fold_train_s1 = set(unique_s1_sorted[train_idx])
        fold_val_s1 = set(unique_s1_sorted[val_idx])

        train_mask = np.fromiter((eid in fold_train_s1 for eid in s1_ids), dtype=bool, count=len(s1_ids))
        val_mask = np.fromiter((eid in fold_val_s1 for eid in s1_ids), dtype=bool, count=len(s1_ids))

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]

        print(f"\n--- Fold {fold}/5 ---")
        print(f"Train set: {len(X_tr):,} pairs (Positives: {int(np.sum(y_tr)):,})")
        print(f"Val set  : {len(X_va):,} pairs (Positives: {int(np.sum(y_va)):,})")

        model_save_path = MODELS_DIR / f"catboost_matcher_fold{fold}.cbm"
        model, metrics = train_catboost_matcher(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_va,
            y_val=y_va,
            iterations=2000,
            learning_rate=0.06,
            depth=8,
            l2_leaf_reg=3.0,
            thread_count=8,
            model_save_path=model_save_path,
            verbose=200,
            random_seed=42 + fold,
        )

        val_p = model.predict_proba(X_va)[:, 1]
        oof_probs[val_mask] = val_p
        models_list.append(model)

        fi_df = get_feature_importances(model, FEATURE_NAMES)
        feature_importances_list.append(fi_df)

        print(f"Fold {fold} training completed in {(time.time() - t0)/60:.2f} mins (best iter: {metrics['best_iteration']})")

    # Save averaged feature importances across 5 folds
    mean_fi = pd.concat(feature_importances_list).groupby("feature", as_index=False)["importance"].mean()
    mean_fi = mean_fi.sort_values(by="importance", ascending=False).reset_index(drop=True)
    print("\nMean Feature Importances Across 5 Folds:")
    print(mean_fi.to_string(index=False))
    mean_fi.to_csv(DIAG_DIR / "feature_importance_v1.csv", index=False)

    # 3. Sweep Thresholds on OOF Predictions
    print("\n" + "=" * 75)
    print("THRESHOLD OPTIMIZATION ON 5-FOLD OOF PREDICTIONS")
    print("=" * 75)

    sweep_df, best_threshold = sweep_thresholds(
        s1_ids=s1_ids,
        candidate_ids=cand_ids,
        probabilities=oof_probs,
        ground_truth_by_s1=gt_dict,
        val_s1_ids=set(unique_s1_sorted),
    )

    print("\nOOF Threshold Sweep Table:")
    print(sweep_df.to_string(index=False))
    sweep_df.to_csv(DIAG_DIR / "threshold_sweep_v1.csv", index=False)

    best_row = sweep_df.loc[sweep_df["threshold"] == best_threshold].iloc[0]

    # Evaluate fold variation at best threshold
    fold_precisions, fold_recalls, fold_f05s = [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_s1_sorted), 1):
        fold_val_s1 = set(unique_s1_sorted[val_idx])
        f_val_mask = np.fromiter((eid in fold_val_s1 for eid in s1_ids), dtype=bool, count=len(s1_ids))
        f_df, _ = sweep_thresholds(
            s1_ids=s1_ids[f_val_mask],
            candidate_ids=cand_ids[f_val_mask],
            probabilities=oof_probs[f_val_mask],
            ground_truth_by_s1=gt_dict,
            val_s1_ids=fold_val_s1,
            thresholds=[best_threshold],
        )
        r = f_df.iloc[0]
        fold_precisions.append(r["macro_precision"])
        fold_recalls.append(r["macro_recall"])
        fold_f05s.append(r["macro_f05"])

    print(f"\nOPTIMAL OOF THRESHOLD: {best_threshold:.2f}")
    print(f"OOF Validation Macro F0.5: {np.mean(fold_f05s):.4f} ± {np.std(fold_f05s):.4f}")
    print(f"OOF Validation Macro P   : {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
    print(f"OOF Validation Macro R   : {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")

    # Free sample arrays memory (keep models in models_list)
    del X, y, oof_probs
    gc.collect()

    # 4. Streaming Full Inference Across All Features with 5-Fold Model Ensemble
    print("\n" + "=" * 75)
    print(f"FULL ENSEMBLE INFERENCE ACROSS ALL CANDIDATES (5 Models, Threshold = {best_threshold:.2f})")
    print("=" * 75)

    all_predictions_by_s1 = defaultdict(set)

    for src_name, src_dir in [("s2", s2_features_dir), ("s3", s3_features_dir)]:
        if not src_dir.exists():
            continue

        p_files = sorted(list(src_dir.glob("part_*.parquet")))
        print(f"\nScoring {len(p_files)} feature chunks for {src_name.upper()} with 5-fold ensemble...")
        t_src_start = time.time()
        matched_pairs_s1 = []
        matched_pairs_cand = []
        matched_probs = []

        for p_idx, p_file in enumerate(p_files):
            df_part = pd.read_parquet(p_file)
            X_part = df_part[FEATURE_NAMES].values.astype(np.float32)

            # Average probabilities across all 5 fold models
            fold_probs = [m.predict_proba(X_part)[:, 1] for m in models_list]
            probs = np.mean(fold_probs, axis=0)

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

    # 5. Overall Full Dataset Macro F0.5 Evaluation
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
        "cv_val_macro_f05_mean": float(np.mean(fold_f05s)),
        "cv_val_macro_f05_std": float(np.std(fold_f05s)),
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
