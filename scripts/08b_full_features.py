"""
Task 8B: Full-Scale Feature Engineering & Checkpointing

Processes Task 7 candidate pairs in chunks, computes pairwise features,
labels candidates using train_ground_truth.tsv, and saves feature chunks.

Supports automatic resume.
"""

from pathlib import Path
import sys
import os
import gc
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.matching import (
    FEATURE_NAMES,
    compute_features_batch,
)

from business_entity_resolution.preprocessing.normalization import (
    load_normalized_or_compute,
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_DIR = Path(
    os.environ.get(
        "BER_DATA_DIR",
        REPO_ROOT / "dataset" / "student_resource" / "dataset" / "train",
    )
)

BLOCKING_DIR = (
    REPO_ROOT
    / "data"
    / "student_resource"
    / "outputs"
    / "blocking"
)

OUTPUT_BASE = (
    REPO_ROOT
    / "data"
    / "student_resource"
    / "outputs"
    / "matching"
    / "features"
)

CHUNK_SIZE = 1_000_000


# ============================================================
# BUILD LOOKUP DICTIONARIES
# ============================================================

def build_lookup_table(df_norm):
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


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth_pairs(gt_file, target_prefix):
    """
    Loads ground-truth pairs for S2 or S3.

    Actual GT columns:
        source1_entity_id
        matched_entity_ids

    Example:
        S1-965667
        S2-681193310,S2-743505751,S3-775321672
    """

    print(f"Loading ground truth for {target_prefix}...")

    gt_pairs = set()

    total_rows = 0
    matched_pairs = 0

    for chunk in pd.read_csv(
        gt_file,
        sep="\t",
        usecols=["source1_entity_id", "matched_entity_ids"],
        dtype=str,
        chunksize=250_000,
    ):
        chunk = chunk.fillna("")

        for s1_id, matched_ids in zip(
            chunk["source1_entity_id"],
            chunk["matched_entity_ids"],
        ):
            total_rows += 1

            if not matched_ids:
                continue

            for target_id in matched_ids.split(","):
                target_id = target_id.strip()

                if target_id.startswith(target_prefix + "-"):
                    gt_pairs.add((s1_id, target_id))
                    matched_pairs += 1

    print(
        f"Ground truth {target_prefix}: "
        f"{len(gt_pairs):,} pairs "
        f"from {total_rows:,} S1 entities"
    )

    return gt_pairs


# ============================================================
# LABEL CANDIDATES
# ============================================================

def label_candidates(s1_ids, candidate_ids, gt_pairs):
    """
    Returns 1 if candidate pair exists in ground truth, else 0.
    """

    return np.fromiter(
        (
            1 if (s1_id, candidate_id) in gt_pairs else 0
            for s1_id, candidate_id in zip(s1_ids, candidate_ids)
        ),
        dtype=np.int8,
        count=len(s1_ids),
    )


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_features_for_source(
    source_name,
    candidate_file,
    output_dir,
    s1_lookup,
    target_lookup,
    gt_pairs,
):
    print()
    print("=" * 80)
    print(f"PROCESSING {source_name}")
    print("=" * 80)

    print(f"Candidate file : {candidate_file}")
    print(f"Output dir     : {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(candidate_file)

    total_rows = parquet_file.metadata.num_rows

    print(f"Candidate rows : {total_rows:,}")
    print(f"Chunk size     : {CHUNK_SIZE:,}")
    print(f"Features       : {len(FEATURE_NAMES)}")

    # --------------------------------------------------------
    # Existing chunks
    # --------------------------------------------------------

    existing_parts = {
        p.name
        for p in output_dir.glob("part_*.parquet")
    }

    if existing_parts:
        print(
            f"Found {len(existing_parts):,} existing chunks. "
            f"Resuming..."
        )

    # --------------------------------------------------------
    # Process chunks
    # --------------------------------------------------------

    start_time = time.time()
    processed_rows = 0
    total_positive = 0

    batch_number = 0

    for batch in parquet_file.iter_batches(
        batch_size=CHUNK_SIZE,
        columns=[
            "s1_entity_id",
            "candidate_entity_id",
            "blocking_passes",
        ],
    ):

        batch_start = time.time()

        part_name = f"part_{batch_number:04d}.parquet"
        output_file = output_dir / part_name

        batch_number += 1

        # ----------------------------------------------------
        # Resume
        # ----------------------------------------------------

        if part_name in existing_parts:
            processed_rows += batch.num_rows

            print(
                f"[SKIP] {part_name} "
                f"({processed_rows:,}/{total_rows:,})"
            )

            continue

        # ----------------------------------------------------
        # Convert Arrow batch
        # ----------------------------------------------------

        data = batch.to_pydict()

        s1_ids = data["s1_entity_id"]
        candidate_ids = data["candidate_entity_id"]
        blocking_passes = data["blocking_passes"]

        n = len(s1_ids)

        # ----------------------------------------------------
        # Build feature inputs
        # ----------------------------------------------------

        s1_names = []
        s1_clean_legal = []
        s1_addresses = []
        s1_countries = []

        target_names = []
        target_clean_legal = []
        target_addresses = []
        target_countries = []

        for s1_id, candidate_id in zip(
            s1_ids,
            candidate_ids,
        ):

            s1_values = s1_lookup.get(
                s1_id,
                ("", "", "", ""),
            )

            target_values = target_lookup.get(
                candidate_id,
                ("", "", "", ""),
            )

            s1_names.append(s1_values[0])
            s1_clean_legal.append(s1_values[1])
            s1_addresses.append(s1_values[2])
            s1_countries.append(s1_values[3])

            target_names.append(target_values[0])
            target_clean_legal.append(target_values[1])
            target_addresses.append(target_values[2])
            target_countries.append(target_values[3])

        # ----------------------------------------------------
        # Compute features
        # ----------------------------------------------------

        features = compute_features_batch(
            s1_names,
            s1_clean_legal,
            s1_addresses,
            s1_countries,
            target_names,
            target_clean_legal,
            target_addresses,
            target_countries,
            blocking_passes,
        )

        # ----------------------------------------------------
        # Convert features to DataFrame
        # ----------------------------------------------------

        if isinstance(features, pd.DataFrame):
            feature_df = features

        else:
            feature_df = pd.DataFrame(
                features,
                columns=FEATURE_NAMES,
            )

        # Make sure column names are correct
        if list(feature_df.columns) != list(FEATURE_NAMES):
            feature_df.columns = FEATURE_NAMES

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        labels = label_candidates(
            s1_ids,
            candidate_ids,
            gt_pairs,
        )

        positive_count = int(labels.sum())

        total_positive += positive_count

        # ----------------------------------------------------
        # Final output dataframe
        # ----------------------------------------------------

        output_df = feature_df.copy()

        output_df.insert(
            0,
            "candidate_entity_id",
            candidate_ids,
        )

        output_df.insert(
            0,
            "s1_entity_id",
            s1_ids,
        )

        output_df.insert(
            0,
            "label",
            labels,
        )

        # ----------------------------------------------------
        # Save Parquet
        # ----------------------------------------------------

        table = pa.Table.from_pandas(
            output_df,
            preserve_index=False,
        )

        pq.write_table(
            table,
            output_file,
            compression="snappy",
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        processed_rows += n

        elapsed = time.time() - start_time

        rate = (
            processed_rows / elapsed
            if elapsed > 0
            else 0
        )

        remaining = total_rows - processed_rows

        eta = (
            remaining / rate
            if rate > 0
            else 0
        )

        print(
            f"[{processed_rows:,}/{total_rows:,}] "
            f"{processed_rows / total_rows * 100:.2f}% | "
            f"positives={positive_count:,} | "
            f"rate={rate:,.0f} rows/s | "
            f"ETA={eta / 60:.1f} min | "
            f"{part_name}"
        )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        del data
        del feature_df
        del output_df
        del table

        del s1_names
        del s1_clean_legal
        del s1_addresses
        del s1_countries

        del target_names
        del target_clean_legal
        del target_addresses
        del target_countries

        gc.collect()

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print()
    print(f"{source_name} COMPLETE")
    print(f"Rows processed : {processed_rows:,}")
    print(f"Positive pairs : {total_positive:,}")
    print(f"Time           : {elapsed / 60:.2f} minutes")
    print(f"Output         : {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TASK 8B — FULL FEATURE ENGINEERING")
    print("=" * 80)

    print()
    print(f"Train Dataset Dir: {TRAIN_DIR}")
    print(f"Blocking Outputs : {BLOCKING_DIR}")
    print(f"Feature Base Dir : {OUTPUT_BASE}")

    # --------------------------------------------------------
    # Load normalized caches
    # --------------------------------------------------------

    print()
    print("Loading normalized cache...")

    columns = [
        "entity_id",
        "name_norm",
        "name_clean_legal",
        "address_norm",
        "country_norm",
    ]

    s1_norm, s2_norm, s3_norm = load_normalized_or_compute(
        TRAIN_DIR,
        REPO_ROOT,
        columns=columns,
    )

    # --------------------------------------------------------
    # Build lookup dictionaries
    # --------------------------------------------------------

    print()
    print("Building lookup dictionaries...")

    s1_lookup = build_lookup_table(s1_norm)
    s2_lookup = build_lookup_table(s2_norm)
    s3_lookup = build_lookup_table(s3_norm)

    print(f"S1 lookup: {len(s1_lookup):,}")
    print(f"S2 lookup: {len(s2_lookup):,}")
    print(f"S3 lookup: {len(s3_lookup):,}")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt_file = TRAIN_DIR / "train_ground_truth.tsv"

    print()
    print(f"Ground truth file: {gt_file}")

    gt_pairs_s2 = load_ground_truth_pairs(
        gt_file,
        target_prefix="S2",
    )

    gt_pairs_s3 = load_ground_truth_pairs(
        gt_file,
        target_prefix="S3",
    )

    # --------------------------------------------------------
    # S1 -> S2
    # --------------------------------------------------------

    s2_candidates = (
        BLOCKING_DIR / "s1_s2_candidates.parquet"
    )

    s2_output = (
        OUTPUT_BASE / "s1_s2"
    )

    process_features_for_source(
        source_name="S1 -> S2",
        candidate_file=s2_candidates,
        output_dir=s2_output,
        s1_lookup=s1_lookup,
        target_lookup=s2_lookup,
        gt_pairs=gt_pairs_s2,
    )

    # --------------------------------------------------------
    # S1 -> S3
    # --------------------------------------------------------

    s3_candidates = (
        BLOCKING_DIR / "s1_s3_candidates.parquet"
    )

    s3_output = (
        OUTPUT_BASE / "s1_s3"
    )

    process_features_for_source(
        source_name="S1 -> S3",
        candidate_file=s3_candidates,
        output_dir=s3_output,
        s1_lookup=s1_lookup,
        target_lookup=s3_lookup,
        gt_pairs=gt_pairs_s3,
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TASK 8B COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()