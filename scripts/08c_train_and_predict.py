"""
Script 08c: CatBoost Training, Full-Candidate OOF Threshold Optimization
and 5-Model Ensemble Inference (Task 8C).

Pipeline:

1. Samples training data:
   - 100% positives
   - hard negatives
   - easy negatives

2. Performs 5-fold entity-level cross-validation using unique S1 entities.

3. Trains 5 CatBoost models.

4. Performs FULL-CANDIDATE OOF inference:
   - Every candidate pair in the 8B feature files is scored.
   - Each S1 entity is scored only by the model for which that entity
     belongs to the validation fold.

5. Optimizes ONE global threshold using the full candidate population.

6. Performs final streaming inference:
   - Every candidate pair is scored by all 5 models.
   - Probabilities are averaged.
   - The optimized global threshold is applied.

7. Computes final Macro F0.5 on the full training candidate population.

8. Saves:
   - 5 CatBoost models
   - predictions
   - feature importance
   - threshold diagnostics
   - metrics
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


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.matching import (
    FEATURE_NAMES,
    load_ground_truth_dict,
    train_catboost_matcher,
    get_feature_importances,
    compute_entity_level_metrics,
)


OUTPUT_BASE = (
    REPO_ROOT
    / "data"
    / "student_resource"
    / "outputs"
    / "matching"
)

FEATURES_DIR = OUTPUT_BASE / "features"
MODELS_DIR = OUTPUT_BASE / "models"
PREDS_DIR = OUTPUT_BASE / "predictions"
DIAG_DIR = OUTPUT_BASE / "diagnostics"

for d in [MODELS_DIR, PREDS_DIR, DIAG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

N_FOLDS = 5

HARD_NEG_RATIO = 3.0
EASY_NEG_RATIO = 1.0

CATBOOST_ITERATIONS = 2000
CATBOOST_LEARNING_RATE = 0.06
CATBOOST_DEPTH = 8
CATBOOST_L2 = 3.0
CATBOOST_THREADS = 8

# Chunk size is determined by 08B parquet files.
# We process one parquet part at a time, so memory stays bounded.

# Threshold search.
#
# First do a coarse search over the whole useful probability range.
COARSE_THRESHOLDS = np.round(
    np.arange(0.50, 1.00, 0.05),
    2,
)

# After finding the best coarse threshold, perform a fine search
# around it.
FINE_STEP = 0.01
FINE_RADIUS = 0.05


# ============================================================
# DATASET LOCATION
# ============================================================

def find_dataset_dir() -> Path:

    p = os.environ.get("BER_DATA_DIR")

    if p and Path(p).is_dir():
        return Path(p)

    candidates = [
        REPO_ROOT
        / "dataset"
        / "student_resource"
        / "dataset"
        / "train",

        REPO_ROOT
        / "data"
        / "student_resource"
        / "dataset"
        / "train",

        Path.home()
        / "Amazon-ML-Hackathon"
        / "dataset"
        / "student_resource"
        / "dataset"
        / "train",
    ]

    for c in candidates:
        if c.is_dir():
            return c

    return (
        REPO_ROOT
        / "dataset"
        / "student_resource"
        / "dataset"
        / "train"
    )


# ============================================================
# TRAINING SAMPLE COLLECTION
# ============================================================

def collect_training_samples(
    source_feature_dirs: List[Path],
    hard_neg_ratio: float = HARD_NEG_RATIO,
    easy_neg_ratio: float = EASY_NEG_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    print("=" * 75)
    print("COLLECTING BALANCED TRAINING DATA FROM FEATURE PARQUETS")
    print("=" * 75)

    pos_dfs = []
    hard_neg_dfs = []
    easy_neg_dfs = []

    rng = np.random.RandomState(random_seed)

    for src_dir in source_feature_dirs:

        parquet_files = sorted(
            list(src_dir.glob("part_*.parquet"))
        )

        print(
            f"Reading {len(parquet_files)} chunks "
            f"from {src_dir.name}..."
        )

        for p_file in parquet_files:

            df_chunk = pd.read_parquet(p_file)

            # ------------------------------------------------
            # Positives
            # ------------------------------------------------

            df_pos = df_chunk[
                df_chunk["label"] == 1
            ]

            if len(df_pos) > 0:
                pos_dfs.append(df_pos)

            n_pos = len(df_pos)

            if n_pos == 0:
                continue

            # ------------------------------------------------
            # Negatives
            # ------------------------------------------------

            df_neg = df_chunk[
                df_chunk["label"] == 0
            ]

            # Hard negatives:
            # reasonably similar names OR multiple blocking passes
            hard_mask = (
                (df_neg["name_token_sort_sim"] >= 0.50)
                |
                (df_neg["name_jaro_winkler"] >= 0.70)
                |
                (df_neg["num_blocking_passes"] >= 2)
            )

            df_hard = df_neg[hard_mask]
            df_easy = df_neg[~hard_mask]

            # ------------------------------------------------
            # Hard negative sampling
            # ------------------------------------------------

            n_hard_sample = min(
                len(df_hard),
                int(n_pos * hard_neg_ratio),
            )

            if n_hard_sample > 0:

                hard_neg_dfs.append(
                    df_hard.sample(
                        n=n_hard_sample,
                        random_state=rng.randint(
                            0,
                            1000000,
                        ),
                    )
                )

            # ------------------------------------------------
            # Easy negative sampling
            # ------------------------------------------------

            n_easy_sample = min(
                len(df_easy),
                int(n_pos * easy_neg_ratio),
            )

            if n_easy_sample > 0:

                easy_neg_dfs.append(
                    df_easy.sample(
                        n=n_easy_sample,
                        random_state=rng.randint(
                            0,
                            1000000,
                        ),
                    )
                )

    print("\nConcatenating sampled dataframes...")

    df_all_pos = pd.concat(
        pos_dfs,
        ignore_index=True,
    )

    df_all_hard = pd.concat(
        hard_neg_dfs,
        ignore_index=True,
    )

    df_all_easy = pd.concat(
        easy_neg_dfs,
        ignore_index=True,
    )

    print(
        f"Total Positives: "
        f"{len(df_all_pos):,}"
    )

    print(
        f"Total Hard Negs: "
        f"{len(df_all_hard):,}"
    )

    print(
        f"Total Easy Negs: "
        f"{len(df_all_easy):,}"
    )

    df_train_full = pd.concat(
        [
            df_all_pos,
            df_all_hard,
            df_all_easy,
        ],
        ignore_index=True,
    )

    df_train_full = (
        df_train_full
        .sample(
            frac=1.0,
            random_state=random_seed,
        )
        .reset_index(drop=True)
    )

    s1_ids = df_train_full[
        "s1_entity_id"
    ].values

    candidate_ids = df_train_full[
        "candidate_entity_id"
    ].values

    labels = (
        df_train_full["label"]
        .values
        .astype(np.int8)
    )

    X = (
        df_train_full[FEATURE_NAMES]
        .values
        .astype(np.float32)
    )

    print(
        f"\nFinal sampled training pool: "
        f"{len(df_train_full):,} pairs "
        f"({np.sum(labels):,} positive)"
    )

    return (
        s1_ids,
        candidate_ids,
        X,
        labels,
    )


# ============================================================
# CREATE ENTITY -> FOLD MAPPING
# ============================================================

def create_entity_fold_mapping(
    unique_s1: np.ndarray,
    n_folds: int = N_FOLDS,
) -> Dict:

    print("\nCreating entity-level fold mapping...")

    kf = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    entity_to_fold = {}

    for fold, (_, val_idx) in enumerate(
        kf.split(unique_s1),
        1,
    ):

        val_entities = unique_s1[val_idx]

        for entity_id in val_entities:
            entity_to_fold[entity_id] = fold

        print(
            f"Fold {fold}: "
            f"{len(val_entities):,} validation S1 entities"
        )

    print(
        f"Mapped {len(entity_to_fold):,} "
        f"unique S1 entities to folds."
    )

    return entity_to_fold


# ============================================================
# BOOLEAN MASK FROM ENTITY FOLD
# ============================================================

def make_fold_mask(
    s1_ids: np.ndarray,
    entity_to_fold: Dict,
    target_fold: int,
) -> np.ndarray:

    return np.fromiter(
        (
            entity_to_fold[eid] == target_fold
            for eid in s1_ids
        ),
        dtype=bool,
        count=len(s1_ids),
    )


# ============================================================
# TRAIN 5 CATBOOST MODELS
# ============================================================

def train_five_models(
    s1_ids: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    unique_s1: np.ndarray,
    entity_to_fold: Dict,
):

    print("\n" + "=" * 75)
    print("5-FOLD ENTITY-LEVEL CROSS-VALIDATION & MODEL TRAINING")
    print("=" * 75)

    oof_probs_sampled = np.zeros(
        len(s1_ids),
        dtype=np.float32,
    )

    models = []
    feature_importances = []

    for fold in range(1, N_FOLDS + 1):

        t0 = time.time()

        val_mask = make_fold_mask(
            s1_ids,
            entity_to_fold,
            fold,
        )

        train_mask = ~val_mask

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_val = X[val_mask]
        y_val = y[val_mask]

        print(f"\n--- Fold {fold}/{N_FOLDS} ---")

        print(
            f"Train set: {len(X_train):,} pairs "
            f"(Positives: {int(np.sum(y_train)):,})"
        )

        print(
            f"Val set  : {len(X_val):,} pairs "
            f"(Positives: {int(np.sum(y_val)):,})"
        )

        model_save_path = (
            MODELS_DIR
            / f"catboost_matcher_fold{fold}.cbm"
        )

        model, metrics = train_catboost_matcher(

            X_train=X_train,
            y_train=y_train,

            X_val=X_val,
            y_val=y_val,

            iterations=CATBOOST_ITERATIONS,

            learning_rate=CATBOOST_LEARNING_RATE,

            depth=CATBOOST_DEPTH,

            l2_leaf_reg=CATBOOST_L2,

            thread_count=CATBOOST_THREADS,

            model_save_path=model_save_path,

            verbose=200,

            random_seed=RANDOM_SEED + fold,
        )

        # OOF predictions for sampled training pool.
        val_probs = (
            model
            .predict_proba(X_val)[:, 1]
        )

        oof_probs_sampled[val_mask] = (
            val_probs.astype(np.float32)
        )

        models.append(model)

        fi_df = get_feature_importances(
            model,
            FEATURE_NAMES,
        )

        feature_importances.append(fi_df)

        elapsed = (
            time.time() - t0
        ) / 60

        print(
            f"Fold {fold} completed in "
            f"{elapsed:.2f} mins "
            f"(best iter: "
            f"{metrics['best_iteration']})"
        )

        del (
            X_train,
            y_train,
            X_val,
            y_val,
            val_probs,
        )

        gc.collect()

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    mean_fi = (
        pd.concat(feature_importances)
        .groupby(
            "feature",
            as_index=False,
        )["importance"]
        .mean()
        .sort_values(
            by="importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\nMean Feature Importances Across 5 Folds:")
    print(
        mean_fi.to_string(index=False)
    )

    mean_fi.to_csv(
        DIAG_DIR
        / "feature_importance_v1.csv",
        index=False,
    )

    return (
        models,
        oof_probs_sampled,
    )


# ============================================================
# FULL-CANDIDATE OOF THRESHOLD EVALUATION
# ============================================================

def evaluate_thresholds_streaming(
    feature_dirs: List[Tuple[str, Path]],
    models: List,
    entity_to_fold: Dict,
    ground_truth_by_s1: Dict,
    thresholds: np.ndarray,
):
    """
    Performs OOF inference over ALL candidate feature rows.

    For every candidate pair:

        S1 entity -> assigned fold -> corresponding model

    This prevents training leakage.

    Instead of storing 264M probabilities, this function
    directly accumulates:

        - predicted count per S1 entity
        - true-positive count per S1 entity

    for each threshold.
    """

    print("\n" + "=" * 75)
    print("FULL-CANDIDATE OOF INFERENCE")
    print("=" * 75)

    thresholds = np.asarray(
        thresholds,
        dtype=np.float32,
    )

    n_thresholds = len(thresholds)

    # --------------------------------------------------------
    # Map every S1 entity to a compact integer index.
    # --------------------------------------------------------

    all_entities = np.array(
        list(ground_truth_by_s1.keys())
    )

    entity_to_index = {
        eid: idx
        for idx, eid in enumerate(all_entities)
    }

    n_entities = len(all_entities)

    # --------------------------------------------------------
    # Arrays:
    #
    # predicted_counts[t, entity]
    # true_positive_counts[t, entity]
    #
    # uint32 is sufficient for candidate counts.
    # --------------------------------------------------------

    predicted_counts = np.zeros(
        (n_thresholds, n_entities),
        dtype=np.uint32,
    )

    true_positive_counts = np.zeros(
        (n_thresholds, n_entities),
        dtype=np.uint32,
    )

    # Ground-truth cardinality for every entity.
    gt_counts = np.zeros(
        n_entities,
        dtype=np.uint32,
    )

    for eid, idx in entity_to_index.items():

        gt_counts[idx] = len(
            ground_truth_by_s1.get(
                eid,
                set(),
            )
        )

    total_candidates = 0

    total_gt_found = 0

    # --------------------------------------------------------
    # Process S2 and S3 separately.
    # --------------------------------------------------------

    for src_name, src_dir in feature_dirs:

        parquet_files = sorted(
            list(src_dir.glob("part_*.parquet"))
        )

        print(
            f"\nOOF scoring {src_name.upper()}: "
            f"{len(parquet_files)} feature chunks"
        )

        source_start = time.time()

        for p_idx, p_file in enumerate(
            parquet_files,
            1,
        ):

            df_part = pd.read_parquet(
                p_file
            )

            s1_part = (
                df_part["s1_entity_id"]
                .values
            )

            candidate_part = (
                df_part["candidate_entity_id"]
                .values
            )

            X_part = (
                df_part[FEATURE_NAMES]
                .values
                .astype(np.float32)
            )

            n_rows = len(df_part)

            total_candidates += n_rows

            # ------------------------------------------------
            # Determine which fold model each row must use.
            # ------------------------------------------------

            fold_numbers = np.fromiter(
                (
                    entity_to_fold.get(
                        eid,
                        0,
                    )
                    for eid in s1_part
                ),
                dtype=np.int8,
                count=n_rows,
            )

            # ------------------------------------------------
            # Score each fold's validation entities with
            # ONLY that fold's model.
            # ------------------------------------------------

            probabilities = np.zeros(
                n_rows,
                dtype=np.float32,
            )

            for fold in range(
                1,
                N_FOLDS + 1,
            ):

                fold_mask = (
                    fold_numbers == fold
                )

                if not np.any(fold_mask):
                    continue

                probabilities[
                    fold_mask
                ] = (
                    models[fold - 1]
                    .predict_proba(
                        X_part[fold_mask]
                    )[:, 1]
                    .astype(np.float32)
                )

            # ------------------------------------------------
            # Convert S1 IDs to compact indexes.
            # ------------------------------------------------

            entity_indices = np.fromiter(
                (
                    entity_to_index.get(
                        eid,
                        -1,
                    )
                    for eid in s1_part
                ),
                dtype=np.int64,
                count=n_rows,
            )

            valid_mask = (
                entity_indices >= 0
            )

            if not np.any(valid_mask):
                del (
                    df_part,
                    X_part,
                    probabilities,
                    fold_numbers,
                    entity_indices,
                )
                continue

            valid_probs = probabilities[
                valid_mask
            ]

            valid_entities = entity_indices[
                valid_mask
            ]

            valid_candidates = candidate_part[
                valid_mask
            ]

            # ------------------------------------------------
            # Find which candidate pairs are actual GT pairs.
            #
            # This is done once per chunk.
            # ------------------------------------------------

            gt_flags = np.zeros(
                len(valid_probs),
                dtype=bool,
            )

            valid_s1_original = s1_part[
                valid_mask
            ]

            for i, (eid, cid) in enumerate(
                zip(
                    valid_s1_original,
                    valid_candidates,
                )
            ):

                gt_set = ground_truth_by_s1.get(
                    eid,
                    set(),
                )

                if cid in gt_set:
                    gt_flags[i] = True

            total_gt_found += int(
                np.sum(gt_flags)
            )

            # ------------------------------------------------
            # Accumulate threshold statistics.
            # ------------------------------------------------

            for t_idx, threshold in enumerate(
                thresholds
            ):

                keep = (
                    valid_probs >= threshold
                )

                if not np.any(keep):
                    continue

                kept_entities = (
                    valid_entities[keep]
                )

                # Number of predicted matches
                # per S1 entity.
                counts = np.bincount(
                    kept_entities,
                    minlength=n_entities,
                )

                predicted_counts[
                    t_idx
                ] += counts.astype(
                    np.uint32
                )

                # True positives.
                tp_mask = (
                    keep
                    & gt_flags
                )

                if np.any(tp_mask):

                    tp_entities = (
                        valid_entities[tp_mask]
                    )

                    tp_counts = np.bincount(
                        tp_entities,
                        minlength=n_entities,
                    )

                    true_positive_counts[
                        t_idx
                    ] += tp_counts.astype(
                        np.uint32
                    )

            if (
                p_idx % 25 == 0
                or p_idx == len(parquet_files)
            ):

                elapsed = (
                    time.time()
                    - source_start
                ) / 60

                print(
                    f"  [{p_idx:03d}/"
                    f"{len(parquet_files):03d}] "
                    f"Candidates: "
                    f"{total_candidates:,} "
                    f"| time: {elapsed:.2f} min"
                )

            del (
                df_part,
                X_part,
                probabilities,
                fold_numbers,
                entity_indices,
                valid_probs,
                valid_entities,
                valid_candidates,
                gt_flags,
            )

            gc.collect()

    # ========================================================
    # Calculate Macro metrics for every threshold.
    # ========================================================

    print("\nCalculating full-candidate OOF metrics...")

    rows = []

    beta = 0.5
    beta_squared = beta ** 2

    for t_idx, threshold in enumerate(
        thresholds
    ):

        pred_counts = (
            predicted_counts[t_idx]
            .astype(np.float64)
        )

        tp_counts = (
            true_positive_counts[t_idx]
            .astype(np.float64)
        )

        gt_counts_float = (
            gt_counts.astype(np.float64)
        )

        # ----------------------------------------------------
        # Entity-level precision.
        # ----------------------------------------------------

        precision = np.divide(
            tp_counts,
            pred_counts,
            out=np.zeros_like(tp_counts),
            where=pred_counts > 0,
        )

        # ----------------------------------------------------
        # Entity-level recall.
        # ----------------------------------------------------

        recall = np.divide(
            tp_counts,
            gt_counts_float,
            out=np.zeros_like(tp_counts),
            where=gt_counts_float > 0,
        )

        # ----------------------------------------------------
        # F0.5
        # ----------------------------------------------------

        f05 = np.divide(
            (1 + beta_squared)
            * precision
            * recall,

            beta_squared * precision
            + recall,

            out=np.zeros_like(precision),

            where=(
                beta_squared * precision
                + recall
            ) > 0,
        )

        macro_precision = float(
            np.mean(precision)
        )

        macro_recall = float(
            np.mean(recall)
        )

        macro_f05 = float(
            np.mean(f05)
        )

        macro_f1 = float(
            np.mean(
                np.divide(
                    2 * precision * recall,
                    precision + recall,
                    out=np.zeros_like(
                        precision
                    ),
                    where=(
                        precision + recall
                    ) > 0,
                )
            )
        )

        predicted_entities = int(
            np.sum(pred_counts > 0)
        )

        total_predictions = int(
            np.sum(pred_counts)
        )

        total_tp = int(
            np.sum(tp_counts)
        )

        rows.append(
            {
                "threshold": float(threshold),
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f05": macro_f05,
                "macro_f1": macro_f1,
                "predicted_entities": predicted_entities,
                "total_predictions": total_predictions,
                "total_true_positives": total_tp,
            }
        )

    metrics_df = pd.DataFrame(rows)

    best_idx = metrics_df[
        "macro_f05"
    ].idxmax()

    best_threshold = float(
        metrics_df.loc[
            best_idx,
            "threshold",
        ]
    )

    print(
        f"\nBest threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print(
        f"\nTotal candidate rows processed: "
        f"{total_candidates:,}"
    )

    print(
        f"GT pairs encountered while scanning: "
        f"{total_gt_found:,}"
    )

    del (
        predicted_counts,
        true_positive_counts,
        gt_counts,
    )

    gc.collect()

    return (
        metrics_df,
        best_threshold,
    )


# ============================================================
# FINAL 5-MODEL ENSEMBLE INFERENCE
# ============================================================

def final_ensemble_inference(
    feature_dirs: List[Tuple[str, Path]],
    models: List,
    threshold: float,
):

    print("\n" + "=" * 75)
    print(
        "FINAL FULL-CANDIDATE ENSEMBLE INFERENCE"
    )
    print("=" * 75)

    all_predictions_by_s1 = defaultdict(set)

    for src_name, src_dir in feature_dirs:

        parquet_files = sorted(
            list(src_dir.glob("part_*.parquet"))
        )

        print(
            f"\nScoring {len(parquet_files)} "
            f"feature chunks for {src_name.upper()}"
        )

        t_src_start = time.time()

        matched_s1 = []
        matched_candidate = []
        matched_probability = []

        for p_idx, p_file in enumerate(
            parquet_files,
            1,
        ):

            df_part = pd.read_parquet(
                p_file
            )

            X_part = (
                df_part[FEATURE_NAMES]
                .values
                .astype(np.float32)
            )

            # ------------------------------------------------
            # Ensemble probability.
            # ------------------------------------------------

            probability_sum = np.zeros(
                len(df_part),
                dtype=np.float32,
            )

            for model in models:

                probability_sum += (
                    model
                    .predict_proba(
                        X_part
                    )[:, 1]
                    .astype(np.float32)
                )

            probabilities = (
                probability_sum
                / len(models)
            )

            keep_mask = (
                probabilities >= threshold
            )

            if np.any(keep_mask):

                k_s1 = (
                    df_part[
                        "s1_entity_id"
                    ]
                    .values[keep_mask]
                )

                k_candidate = (
                    df_part[
                        "candidate_entity_id"
                    ]
                    .values[keep_mask]
                )

                k_probability = (
                    probabilities[
                        keep_mask
                    ]
                )

                matched_s1.extend(
                    k_s1
                )

                matched_candidate.extend(
                    k_candidate
                )

                matched_probability.extend(
                    k_probability
                )

                # ------------------------------------------------
                # Entity-level prediction dictionary.
                # ------------------------------------------------

                for s1, candidate in zip(
                    k_s1,
                    k_candidate,
                ):

                    all_predictions_by_s1[
                        s1
                    ].add(candidate)

            if (
                p_idx % 25 == 0
                or p_idx == len(parquet_files)
            ):

                print(
                    f"  [{p_idx:03d}/"
                    f"{len(parquet_files):03d}] "
                    f"Processed. "
                    f"Matches kept so far: "
                    f"{len(matched_s1):,}"
                )

            del (
                df_part,
                X_part,
                probability_sum,
                probabilities,
                keep_mask,
            )

            gc.collect()

        # ----------------------------------------------------
        # Save predictions.
        # ----------------------------------------------------

        df_predictions = pd.DataFrame(
            {
                "s1_entity_id": matched_s1,
                "matched_entity_id": matched_candidate,
                "probability": matched_probability,
            }
        )

        pred_file = (
            PREDS_DIR
            / f"s1_{src_name}_matches.parquet"
        )

        df_predictions.to_parquet(
            pred_file,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        elapsed = (
            time.time()
            - t_src_start
        ) / 60

        print(
            f"Saved {len(df_predictions):,} "
            f"{src_name.upper()} predictions to "
            f"{pred_file.name}"
        )

        print(
            f"Source inference time: "
            f"{elapsed:.2f} mins"
        )

        del (
            df_predictions,
            matched_s1,
            matched_candidate,
            matched_probability,
        )

        gc.collect()

    return all_predictions_by_s1


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    # ========================================================
    # Dataset
    # ========================================================

    train_dir = find_dataset_dir()

    print(
        f"Train Dataset Dir: "
        f"{train_dir}"
    )

    print(
        f"Features Base Dir: "
        f"{FEATURES_DIR}"
    )

    # ========================================================
    # Feature directories
    # ========================================================

    s2_features_dir = (
        FEATURES_DIR / "s1_s2"
    )

    s3_features_dir = (
        FEATURES_DIR / "s1_s3"
    )

    if (
        not s2_features_dir.exists()
        and not s3_features_dir.exists()
    ):

        print(
            f"ERROR: Feature directories "
            f"not found in {FEATURES_DIR}."
        )

        print(
            "Run 08b_full_features.py first."
        )

        return

    feature_dirs = [
        (name, path)
        for name, path in [
            ("s2", s2_features_dir),
            ("s3", s3_features_dir),
        ]
        if path.exists()
    ]

    # ========================================================
    # Ground truth
    # ========================================================

    gt_file = (
        train_dir
        / "train_ground_truth.tsv"
    )

    if not gt_file.is_file():

        alt_gt = (
            REPO_ROOT
            / "dataset"
            / "student_resource"
            / "dataset"
            / "train"
            / "train_ground_truth.tsv"
        )

        if alt_gt.is_file():
            gt_file = alt_gt

    print(
        f"\nLoading ground truth dictionary "
        f"from {gt_file}..."
    )

    gt_dict = load_ground_truth_dict(
        gt_file
    )

    print(
        f"Ground truth S1 entities: "
        f"{len(gt_dict):,}"
    )

    # ========================================================
    # STEP 1
    # Collect training samples
    # ========================================================

    s1_ids, candidate_ids, X, y = (
        collect_training_samples(
            [
                path
                for _, path in feature_dirs
            ]
        )
    )

    # ========================================================
    # STEP 2
    # Entity-level fold assignment
    # ========================================================

    unique_s1 = np.array(
        sorted(
            list(
                set(s1_ids)
            )
        )
    )

    print(
        f"\nTraining pool contains "
        f"{len(unique_s1):,} "
        f"unique S1 entities."
    )

    entity_to_fold = (
        create_entity_fold_mapping(
            unique_s1,
            N_FOLDS,
        )
    )

    # ========================================================
    # STEP 3
    # Train 5 CatBoost models
    # ========================================================

    models, sampled_oof_probs = (
        train_five_models(
            s1_ids=s1_ids,
            X=X,
            y=y,
            unique_s1=unique_s1,
            entity_to_fold=entity_to_fold,
        )
    )

    # ========================================================
    # Sampled OOF diagnostic
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "SAMPLED TRAINING-POOL OOF COMPLETE"
    )
    print("=" * 75)

    print(
        f"Sampled OOF rows: "
        f"{len(sampled_oof_probs):,}"
    )

    # We deliberately DO NOT use these sampled
    # probabilities to select the final threshold.

    del (
        sampled_oof_probs,
        X,
        y,
        s1_ids,
        candidate_ids,
    )

    gc.collect()

    # ========================================================
    # STEP 4
    # Full-candidate OOF threshold optimization
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "GLOBAL THRESHOLD OPTIMIZATION "
        "ON FULL CANDIDATE POPULATION"
    )
    print("=" * 75)

    print(
        "\nThis stage scores all ~264.7M candidate pairs."
    )

    print(
        "Each S1 entity is scored using only its "
        "corresponding validation-fold model."
    )

    # --------------------------------------------------------
    # Coarse threshold search
    # --------------------------------------------------------

    print(
        "\nStarting coarse threshold search:"
    )

    print(
        COARSE_THRESHOLDS
    )

    coarse_df, coarse_best = (
        evaluate_thresholds_streaming(
            feature_dirs=feature_dirs,
            models=models,
            entity_to_fold=entity_to_fold,
            ground_truth_by_s1=gt_dict,
            thresholds=COARSE_THRESHOLDS,
        )
    )

    coarse_df.to_csv(
        DIAG_DIR
        / "threshold_sweep_full_candidate_coarse.csv",
        index=False,
    )

    # ========================================================
    # Fine threshold search
    # ========================================================

    fine_low = max(
        0.50,
        coarse_best - FINE_RADIUS,
    )

    fine_high = min(
        0.99,
        coarse_best + FINE_RADIUS,
    )

    fine_thresholds = np.round(
        np.arange(
            fine_low,
            fine_high + 0.0001,
            FINE_STEP,
        ),
        2,
    )

    print(
        f"\nFine threshold search around "
        f"{coarse_best:.2f}:"
    )

    print(
        fine_thresholds
    )

    fine_df, best_threshold = (
        evaluate_thresholds_streaming(
            feature_dirs=feature_dirs,
            models=models,
            entity_to_fold=entity_to_fold,
            ground_truth_by_s1=gt_dict,
            thresholds=fine_thresholds,
        )
    )

    fine_df.to_csv(
        DIAG_DIR
        / "threshold_sweep_full_candidate_fine.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final threshold
    # --------------------------------------------------------

    best_row = fine_df.loc[
        fine_df["macro_f05"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    print("\n" + "=" * 75)
    print("FINAL GLOBAL THRESHOLD")
    print("=" * 75)

    print(
        f"Threshold       : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Macro Precision : "
        f"{best_row['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall    : "
        f"{best_row['macro_recall']:.4f}"
    )

    print(
        f"Macro F0.5      : "
        f"{best_row['macro_f05']:.4f}"
    )

    print(
        f"Macro F1        : "
        f"{best_row['macro_f1']:.4f}"
    )

    print(
        f"Predicted pairs : "
        f"{int(best_row['total_predictions']):,}"
    )

    # ========================================================
    # STEP 5
    # Final 5-model ensemble inference
    # ========================================================

    all_predictions_by_s1 = (
        final_ensemble_inference(
            feature_dirs=feature_dirs,
            models=models,
            threshold=best_threshold,
        )
    )

    # ========================================================
    # STEP 6
    # Final full-dataset evaluation
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "FINAL FULL DATASET EVALUATION"
    )
    print("=" * 75)

    final_metrics = (
        compute_entity_level_metrics(
            predictions_by_s1=all_predictions_by_s1,
            ground_truth_by_s1=gt_dict,
        )
    )

    print(
        f"Macro Precision : "
        f"{final_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall    : "
        f"{final_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F0.5      : "
        f"{final_metrics['macro_f05']:.4f}"
    )

    print(
        f"Macro F1        : "
        f"{final_metrics['macro_f1']:.4f}"
    )

    print(
        f"Evaluated S1 IDs: "
        f"{final_metrics['num_evaluated_entities']:,}"
    )

    # ========================================================
    # Save metrics
    # ========================================================

    metrics_output = {

        "pipeline": "task_8c",

        "n_folds": N_FOLDS,

        "training_sampling": {
            "positive_ratio": "100%",
            "hard_negative_ratio": HARD_NEG_RATIO,
            "easy_negative_ratio": EASY_NEG_RATIO,
        },

        "threshold_source":
            "full_candidate_oof",

        "best_threshold":
            float(best_threshold),

        "full_candidate_oof_macro_f05":
            float(
                best_row["macro_f05"]
            ),

        "full_candidate_oof_macro_precision":
            float(
                best_row["macro_precision"]
            ),

        "full_candidate_oof_macro_recall":
            float(
                best_row["macro_recall"]
            ),

        "full_ensemble_macro_f05":
            float(
                final_metrics["macro_f05"]
            ),

        "full_ensemble_macro_precision":
            float(
                final_metrics["macro_precision"]
            ),

        "full_ensemble_macro_recall":
            float(
                final_metrics["macro_recall"]
            ),

        "full_ensemble_macro_f1":
            float(
                final_metrics["macro_f1"]
            ),

        "evaluated_entities":
            int(
                final_metrics[
                    "num_evaluated_entities"
                ]
            ),
    }

    metrics_file = (
        MODELS_DIR
        / "metrics_v1.json"
    )

    with open(
        metrics_file,
        "w",
    ) as f:

        json.dump(
            metrics_output,
            f,
            indent=2,
        )

    print(
        f"\nSaved metrics to "
        f"{metrics_file}"
    )

    # ========================================================
    # Final timing
    # ========================================================

    total_minutes = (
        time.time()
        - total_start
    ) / 60

    print("\n" + "=" * 75)

    print(
        f"TASK 8C COMPLETE"
    )

    print(
        f"Total runtime: "
        f"{total_minutes:.2f} minutes "
        f"({total_minutes / 60:.2f} hours)"
    )

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main() 