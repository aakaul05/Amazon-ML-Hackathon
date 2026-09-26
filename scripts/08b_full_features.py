"""
Script 08b: Full-Scale Feature Engineering & Checkpointing (Task 8B).
Extracts, computes 18 pairwise features, labels against ground truth, and checkpoints
all ~264M candidate pairs into compact Parquet chunks (~1M rows per part).
Supports automatic resume if interrupted.
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.matching import (
    FEATURE_NAMES,
    compute_features_batch,
    load_ground_truth_pairs,
    label_candidates_batch,
)
from business_entity_resolution.preprocessing.normalization import (
    load_normalized_or_compute,
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


def find_blocking_dir() -> Path:
    candidates = [
        REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking",
        REPO_ROOT / "data" / "outputs" / "blocking",
        Path.home() / "Amazon-ML-Hackathon" / "data" / "student_resource" / "outputs" / "blocking",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking"


OUTPUT_BASE = REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "features"


def build_lookup_table(df_norm: pd.DataFrame) -> Dict[str, Tuple[str, str, str, str]]:
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


def process_features_for_source(
    target_name: str,
    cand_path: Path,
    out_dir: Path,
    s1_lookup: Dict[str, Tuple[str, str, str, str]],
    target_lookup: Dict[str, Tuple[str, str, str, str]],
    gt_pairs: set,
    chunk_size: int = 1_000_000,
):
    print("=" * 75)
    print(f"FULL FEATURE EXTRACTION: S1 -> {target_name.upper()}")
    print(f"Candidate file : {cand_path}")
    print(f"Output chunk dir: {out_dir}")
    print("=" * 75)

    if not cand_path.is_file():
        print(f"ERROR: Candidate file not found: {cand_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(str(cand_path))
    num_total_rows = parquet_file.metadata.num_rows
    print(f"Total candidate rows: {num_total_rows:,}")

    total_chunks = (num_total_rows + chunk_size - 1) // chunk_size
    print(f"Splitting into {total_chunks} chunks (~{chunk_size:,} rows/chunk)...")

    start_all = time.time()
    chunk_idx = 0
    total_positives = 0
    total_pairs_saved = 0

    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"]):
        out_chunk_path = out_dir / f"part_{chunk_idx:04d}.parquet"

        if out_chunk_path.exists():
            print(f"[{chunk_idx+1:03d}/{total_chunks:03d}] Skipping (already exists): {out_chunk_path.name}")
            chunk_idx += 1
            continue

        t0 = time.time()
        df_batch = batch.to_pandas()
        n_pairs = len(df_batch)

        s1_names = [""] * n_pairs
        s1_clean = [""] * n_pairs
        s1_addrs = [""] * n_pairs
        s1_countries = [""] * n_pairs

        ot_names = [""] * n_pairs
        ot_clean = [""] * n_pairs
        ot_addrs = [""] * n_pairs
        ot_countries = [""] * n_pairs

        s1_eids = df_batch["s1_entity_id"].values
        cand_eids = df_batch["candidate_entity_id"].values
        bp_list = df_batch["blocking_passes"].values

        for i in range(n_pairs):
            s1_id = s1_eids[i]
            cand_id = cand_eids[i]

            s1_data = s1_lookup.get(s1_id)
            if s1_data is not None:
                s1_names[i], s1_clean[i], s1_addrs[i], s1_countries[i] = s1_data

            ot_data = target_lookup.get(cand_id)
            if ot_data is not None:
                ot_names[i], ot_clean[i], ot_addrs[i], ot_countries[i] = ot_data

        # Compute features
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

        # Label against ground truth
        labels = label_candidates_batch(s1_eids, cand_eids, gt_pairs)
        n_pos = int(np.sum(labels))
        total_positives += n_pos

        # Construct DataFrame
        chunk_dict = {
            "s1_entity_id": s1_eids,
            "candidate_entity_id": cand_eids,
            "label": labels,
        }
        for f_idx, f_name in enumerate(FEATURE_NAMES):
            chunk_dict[f_name] = features[:, f_idx]

        out_df = pd.DataFrame(chunk_dict)
        # Write to parquet with snappy compression
        out_df.to_parquet(out_chunk_path, engine="pyarrow", compression="snappy", index=False)

        elapsed = time.time() - t0
        total_elapsed = time.time() - start_all
        total_pairs_saved += n_pairs
        avg_chunk_time = total_elapsed / (chunk_idx + 1)
        remaining_chunks = total_chunks - (chunk_idx + 1)
        eta_sec = remaining_chunks * avg_chunk_time

        print(
            f"[{chunk_idx+1:03d}/{total_chunks:03d}] "
            f"Saved {n_pairs:,} pairs (Pos: {n_pos:,}) in {elapsed:.1f}s | "
            f"Total elapsed: {total_elapsed/60:.1f}m | ETA: {eta_sec/60:.1f}m"
        )

        chunk_idx += 1
        gc.collect()

    print(f"\nFINISHED Feature extraction for S1 -> {target_name.upper()} in {(time.time() - start_all)/60:.2f} mins")
    print(f"Total pairs processed: {total_pairs_saved:,}, Total Positives: {total_positives:,}")


def main():
    train_dir = find_dataset_dir()
    blocking_dir = find_blocking_dir()

    print(f"Train Dataset Dir: {train_dir}")
    print(f"Blocking Outputs : {blocking_dir}")
    print(f"Feature Base Dir : {OUTPUT_BASE}")

    print("\nLoading normalized cache...")
    s1_norm = load_normalized_or_compute(train_dir, "s1", columns=["entity_id", "name_norm", "name_clean_legal", "address_norm", "country_norm"])
    s2_norm = load_normalized_or_compute(train_dir, "s2", columns=["entity_id", "name_norm", "name_clean_legal", "address_norm", "country_norm"])
    s3_norm = load_normalized_or_compute(train_dir, "s3", columns=["entity_id", "name_norm", "name_clean_legal", "address_norm", "country_norm"])

    print("Building lookup dictionaries...")
    s1_lookup = build_lookup_table(s1_norm)
    s2_lookup = build_lookup_table(s2_norm)
    s3_lookup = build_lookup_table(s3_norm)

    # Load ground truth
    gt_file = train_dir / "train_ground_truth.tsv"
    if not gt_file.is_file():
        alt_gt = REPO_ROOT / "dataset" / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"
        if alt_gt.is_file():
            gt_file = alt_gt

    print(f"Loading ground truth from {gt_file}...")
    gt_pairs_s2 = load_ground_truth_pairs(gt_file, target_prefix="S2")
    gt_pairs_s3 = load_ground_truth_pairs(gt_file, target_prefix="S3")
    print(f"Loaded {len(gt_pairs_s2):,} S2 positive pairs and {len(gt_pairs_s3):,} S3 positive pairs.")

    # Process S1 -> S2
    s2_cand_file = blocking_dir / "s1_s2_candidates.parquet"
    s2_out_dir = OUTPUT_BASE / "s1_s2"
    process_features_for_source("s2", s2_cand_file, s2_out_dir, s1_lookup, s2_lookup, gt_pairs_s2)

    gc.collect()

    # Process S1 -> S3
    s3_cand_file = blocking_dir / "s1_s3_candidates.parquet"
    s3_out_dir = OUTPUT_BASE / "s1_s3"
    process_features_for_source("s3", s3_cand_file, s3_out_dir, s1_lookup, s3_lookup, gt_pairs_s3)


if __name__ == "__main__":
    main()
