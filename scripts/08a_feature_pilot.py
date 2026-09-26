"""
Script 08a: Feature Engineering & Model Training Pilot (Task 8A).

Runs an end-to-end matching pipeline on a ~2M pair sample per source to validate:

- Feature computation correctness & performance
- Label alignment with ground truth
- CatBoost training dynamics & AUC
- Macro F0.5 threshold optimization curve
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from business_entity_resolution.matching import (
    FEATURE_NAMES,
    compute_features_batch,
    label_candidates_batch,
    entity_level_split,
    train_catboost_matcher,
    get_feature_importances,
    sweep_thresholds,
)

from business_entity_resolution.preprocessing.normalization import (
    load_normalized_or_compute,
)


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


def find_blocking_dir() -> Path:
    candidates = [
        REPO_ROOT
        / "data"
        / "student_resource"
        / "outputs"
        / "blocking",

        REPO_ROOT
        / "data"
        / "outputs"
        / "blocking",

        Path.home()
        / "Amazon-ML-Hackathon"
        / "data"
        / "student_resource"
        / "outputs"
        / "blocking",
    ]

    for c in candidates:
        if c.is_dir():
            return c

    return (
        REPO_ROOT
        / "data"
        / "student_resource"
        / "outputs"
        / "blocking"
    )


OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "student_resource"
    / "outputs"
    / "matching"
    / "pilot"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth(
    gt_file_path: Path,
) -> Tuple[Dict[str, Set[str]], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    Loads ground truth using the actual competition schema:

        source1_entity_id    matched_entity_ids

    Returns:
        gt_dict:
            S1 entity ID -> set of all matched S2/S3 entity IDs

        gt_pairs_s2:
            Set of (S1, S2) positive pairs

        gt_pairs_s3:
            Set of (S1, S3) positive pairs
    """

    df = pd.read_csv(
        gt_file_path,
        sep="\t",
        dtype={
            "source1_entity_id": str,
            "matched_entity_ids": str,
        },
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    gt_dict: Dict[str, Set[str]] = {}
    gt_pairs_s2: Set[Tuple[str, str]] = set()
    gt_pairs_s3: Set[Tuple[str, str]] = set()

    for s1_id, matches in zip(
        df["source1_entity_id"],
        df["matched_entity_ids"],
    ):
        if pd.isna(matches) or not str(matches).strip():
            gt_dict[str(s1_id)] = set()
            continue

        match_set = {
            m.strip()
            for m in str(matches).split(",")
            if m.strip()
        }

        gt_dict[str(s1_id)] = match_set

        for match_id in match_set:
            if match_id.startswith("S2"):
                gt_pairs_s2.add(
                    (str(s1_id), match_id)
                )

            elif match_id.startswith("S3"):
                gt_pairs_s3.add(
                    (str(s1_id), match_id)
                )

    return gt_dict, gt_pairs_s2, gt_pairs_s3


def build_lookup_table(
    df_norm: pd.DataFrame,
) -> Dict[str, Tuple[str, str, str, str]]:
    """
    Build fast O(1) lookup:

    entity_id ->
        (
            name_norm,
            name_clean_legal,
            address_norm,
            country_norm
        )
    """

    lookup = {}

    for eid, nn, ncl, an, cn in zip(
        df_norm["entity_id"],
        df_norm["name_norm"],
        df_norm["name_clean_legal"],
        df_norm["address_norm"],
        df_norm["country_norm"],
    ):
        lookup[eid] = (
            nn if isinstance(nn, str) else "",
            ncl if isinstance(ncl, str) else "",
            an if isinstance(an, str) else "",
            cn if isinstance(cn, str) else "",
        )

    return lookup


def run_pilot_for_source(
    target_name: str,
    cand_path: Path,
    s1_lookup: Dict[str, Tuple[str, str, str, str]],
    target_lookup: Dict[str, Tuple[str, str, str, str]],
    gt_pairs: Set[Tuple[str, str]],
    gt_dict: Dict[str, Set[str]],
    sample_size: int = 2_000_000,
):

    print("=" * 70)

    print(
        f"PILOT MATCHING: S1 -> {target_name.upper()} "
        f"(Sample: {sample_size:,} candidate pairs)"
    )

    print("=" * 70)

    if not cand_path.is_file():
        print(
            f"ERROR: Candidate file not found: {cand_path}"
        )
        return

    # ---------------------------------------------------------
    # Read candidate sample
    # ---------------------------------------------------------

    print(
        f"Reading candidate pairs from {cand_path.name}..."
    )

    t0 = time.time()

    parquet_file = pq.ParquetFile(str(cand_path))

    batches = []
    total_read = 0

    for batch in parquet_file.iter_batches(
        batch_size=250_000,
        columns=[
            "s1_entity_id",
            "candidate_entity_id",
            "blocking_passes",
        ],
    ):
        df_b = batch.to_pandas()

        batches.append(df_b)

        total_read += len(df_b)

        if total_read >= sample_size:
            break

    if not batches:
        print("ERROR: Candidate file contains no rows.")
        return

    cand_df = pd.concat(
        batches,
        ignore_index=True,
    ).iloc[:sample_size]

    print(
        f"Loaded {len(cand_df):,} candidate pairs "
        f"in {time.time() - t0:.2f}s"
    )

    del batches
    gc.collect()

    # ---------------------------------------------------------
    # Extract normalized attributes
    # ---------------------------------------------------------

    print(
        "Extracting normalized attributes for candidate pairs..."
    )

    t0 = time.time()

    n_pairs = len(cand_df)

    s1_names = [""] * n_pairs
    s1_clean = [""] * n_pairs
    s1_addrs = [""] * n_pairs
    s1_countries = [""] * n_pairs

    ot_names = [""] * n_pairs
    ot_clean = [""] * n_pairs
    ot_addrs = [""] * n_pairs
    ot_countries = [""] * n_pairs

    s1_eids = cand_df["s1_entity_id"].values
    cand_eids = cand_df["candidate_entity_id"].values
    bp_list = cand_df["blocking_passes"].values

    for i in range(n_pairs):

        s1_id = s1_eids[i]
        cand_id = cand_eids[i]

        s1_data = s1_lookup.get(s1_id)

        if s1_data is not None:
            (
                s1_names[i],
                s1_clean[i],
                s1_addrs[i],
                s1_countries[i],
            ) = s1_data

        ot_data = target_lookup.get(cand_id)

        if ot_data is not None:
            (
                ot_names[i],
                ot_clean[i],
                ot_addrs[i],
                ot_countries[i],
            ) = ot_data

    print(
        f"Extracted attributes in "
        f"{time.time() - t0:.2f}s"
    )

    # ---------------------------------------------------------
    # Compute features
    # ---------------------------------------------------------

    print("Computing 18 pairwise features...")

    t0 = time.time()

    features = compute_features_batch(
        s1_names=s1_names,
        s1_clean=s1_clean,
        s1_addrs=s1_addrs,
        s1_countries=s1_countries,
        ot_names=ot_names,
        ot_clean=ot_clean,
        ot_addrs=ot_addrs,
        ot_countries=ot_countries,
        blocking_passes=bp_list,
    )

    print(
        f"Computed features array of shape "
        f"{features.shape} "
        f"in {time.time() - t0:.2f}s"
    )

    # ---------------------------------------------------------
    # Label candidates
    # ---------------------------------------------------------

    print(
        "Labeling pairs against ground truth..."
    )

    t0 = time.time()

    labels = label_candidates_batch(
        s1_eids,
        cand_eids,
        gt_pairs,
    )

    pos_count = int(np.sum(labels))
    neg_count = len(labels) - pos_count

    pos_rate = (
        pos_count / len(labels) * 100.0
        if len(labels) > 0
        else 0.0
    )

    print(
        f"Labeled in {time.time() - t0:.2f}s: "
        f"{pos_count:,} Positives ({pos_rate:.2f}%), "
        f"{neg_count:,} Negatives"
    )

    # ---------------------------------------------------------
    # Entity-level train/validation split
    # ---------------------------------------------------------

    print(
        "\nPerforming entity-level train/val split (80/20)..."
    )

    train_s1, val_s1 = entity_level_split(
        s1_eids,
        val_fraction=0.20,
        seed=42,
    )

    train_mask = np.fromiter(
        (eid in train_s1 for eid in s1_eids),
        dtype=bool,
        count=len(s1_eids),
    )

    val_mask = ~train_mask

    X_train = features[train_mask]
    y_train = labels[train_mask]

    X_val = features[val_mask]
    y_val = labels[val_mask]

    print(
        f"Train set: {len(X_train):,} pairs "
        f"(Positives: {int(np.sum(y_train)):,})"
    )

    print(
        f"Val set  : {len(X_val):,} pairs "
        f"(Positives: {int(np.sum(y_val)):,})"
    )

    # ---------------------------------------------------------
    # Train CatBoost
    # ---------------------------------------------------------

    print(
        "\nTraining CatBoost Classifier..."
    )

    t0 = time.time()

    model_save_path = (
        OUTPUT_DIR
        / f"catboost_pilot_{target_name}.cbm"
    )

    model, metrics = train_catboost_matcher(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        iterations=1000,
        learning_rate=0.08,
        depth=7,
        thread_count=8,
        model_save_path=model_save_path,
        verbose=100,
    )

    print(
        f"Training completed in "
        f"{time.time() - t0:.2f}s. "
        f"Best iteration: {metrics['best_iteration']}"
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    df_fi = get_feature_importances(
        model,
        FEATURE_NAMES,
    )

    print(
        "\nTop 10 Most Important Features:"
    )

    print(
        df_fi.head(10).to_string(index=False)
    )

    df_fi.to_csv(
        OUTPUT_DIR
        / f"feature_importance_pilot_{target_name}.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Validation predictions + threshold sweep
    # ---------------------------------------------------------

    print(
        "\nEvaluating probabilities and sweeping "
        "decision thresholds for Macro F0.5..."
    )

    val_probs = model.predict_proba(
        X_val
    )[:, 1]

    val_s1_eval = s1_eids[val_mask]
    val_cand_eval = cand_eids[val_mask]

    sweep_df, best_th = sweep_thresholds(
        s1_ids=val_s1_eval,
        candidate_ids=val_cand_eval,
        probabilities=val_probs,
        ground_truth_by_s1=gt_dict,
        val_s1_ids=val_s1,
    )

    print(
        "\nThreshold Sweep Results:"
    )

    print(
        sweep_df.to_string(index=False)
    )

    sweep_df.to_csv(
        OUTPUT_DIR
        / f"threshold_sweep_pilot_{target_name}.csv",
        index=False,
    )

    best_row = sweep_df.loc[
        sweep_df["threshold"] == best_th
    ].iloc[0]

    print("=" * 70)

    print(
        f"PILOT RESULTS FOR S1 -> "
        f"{target_name.upper()}:"
    )

    print(
        f"  Best Threshold   : {best_th:.2f}"
    )

    print(
        f"  Macro F0.5 Score : "
        f"{best_row['macro_f05']:.4f}"
    )

    print(
        f"  Macro Precision  : "
        f"{best_row['macro_precision']:.4f}"
    )

    print(
        f"  Macro Recall     : "
        f"{best_row['macro_recall']:.4f}"
    )

    print(
        f"  Macro F1 Score   : "
        f"{best_row['macro_f1']:.4f}"
    )

    print("=" * 70)

    # ---------------------------------------------------------
    # Free memory before next source
    # ---------------------------------------------------------

    del cand_df
    del features
    del labels
    del X_train
    del y_train
    del X_val
    del y_val
    del val_probs

    gc.collect()


def main():

    train_dir = find_dataset_dir()
    blocking_dir = find_blocking_dir()

    print(
        f"Train Dataset Dir: {train_dir}"
    )

    print(
        f"Blocking Outputs : {blocking_dir}"
    )

    print(
        f"Pilot Outputs    : {OUTPUT_DIR}"
    )

    # ---------------------------------------------------------
    # Load existing normalized cache
    # ---------------------------------------------------------

    print(
        "\nLoading normalized cache..."
    )

    columns = [
        "entity_id",
        "name_norm",
        "name_clean_legal",
        "address_norm",
        "country_norm",
    ]

    # IMPORTANT:
    # load_normalized_or_compute() returns ALL THREE dataframes.
    #
    # Pass REPO_ROOT so it looks in:
    #
    # data/outputs/normalized_cache/

    s1_norm, s2_norm, s3_norm = (
        load_normalized_or_compute(
            train_dir,
            REPO_ROOT,
            columns=columns,
        )
    )

    print(
        "Building lookup dictionaries..."
    )

    s1_lookup = build_lookup_table(
        s1_norm
    )

    s2_lookup = build_lookup_table(
        s2_norm
    )

    s3_lookup = build_lookup_table(
        s3_norm
    )

    # We no longer need the normalized
    # DataFrames after creating lookup dictionaries.

    del s1_norm
    del s2_norm
    del s3_norm

    gc.collect()

    # ---------------------------------------------------------
    # Load ground truth
    # ---------------------------------------------------------

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
        f"Loading ground truth from {gt_file}..."
    )

    gt_dict, gt_pairs_s2, gt_pairs_s3 = (
        load_ground_truth(gt_file)
    )

    print(
        f"Loaded {len(gt_pairs_s2):,} S2 "
        f"positive pairs and "
        f"{len(gt_pairs_s3):,} S3 positive pairs."
    )

    # ---------------------------------------------------------
    # Run pilot for S2
    # ---------------------------------------------------------

    s2_cand_file = (
        blocking_dir
        / "s1_s2_candidates.parquet"
    )

    run_pilot_for_source(
        "s2",
        s2_cand_file,
        s1_lookup,
        s2_lookup,
        gt_pairs_s2,
        gt_dict,
        sample_size=2_000_000,
    )

    gc.collect()

    # ---------------------------------------------------------
    # Run pilot for S3
    # ---------------------------------------------------------

    s3_cand_file = (
        blocking_dir
        / "s1_s3_candidates.parquet"
    )

    run_pilot_for_source(
        "s3",
        s3_cand_file,
        s1_lookup,
        s3_lookup,
        gt_pairs_s3,
        gt_dict,
        sample_size=2_000_000,
    )


if __name__ == "__main__":
    main()