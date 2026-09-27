"""
scripts/11_test_inference.py
============================
Step 3 of Test Pipeline: Streaming 30-Feature Computation & 5-Model Ensemble Inference.

Pipeline Architecture:
  1. Loads all 5 trained CatBoost models (catboost_matcher_fold1..5.cbm).
  2. Builds normalized attribute lookup dictionaries for S1, S2, S3.
  3. Streams test candidate parquet in chunks (~500,000 pairs/chunk).
  4. For each chunk:
     - Extracts attribute strings.
     - Computes exact 30 pairwise features (features.py).
     - Obtains prediction probability from each of the 5 CatBoost models.
     - Computes ensemble probability = (P1 + P2 + P3 + P4 + P5) / 5.0.
     - Retains candidate matches where ensemble probability >= 0.9800000190734863.
     - Releases feature matrix and probability arrays immediately from RAM.
     - Checkpoints kept matches to part_XXXX.parquet.
  5. Concatenates kept match parts into final intermediate prediction parquets:
     - data/student_resource/outputs/test/predictions/test_s1_s2_matches.parquet
     - data/student_resource/outputs/test/predictions/test_s1_s3_matches.parquet

Disk & Memory Safety:
  - NEVER writes huge 200M+ row feature matrices to disk (preserves EC2 50 GB storage).
  - Keeps memory usage bounded within 64 GB RAM.
  - Resumable: resumes from next chunk if interrupted.
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# pyrefly: ignore [missing-import]
from catboost import CatBoostClassifier

from business_entity_resolution.matching.features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    compute_features_batch,
)

NORM_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "test" / "normalized"
BLOCKING_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "test" / "blocking"
PRED_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "test" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def find_models_dir() -> Path:
    env_dir = os.environ.get("BER_MODELS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir() and (p / "catboost_matcher_fold1.cbm").is_file():
            return p.resolve()

    candidates = [
        REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models",
        Path("/home/ec2-user/Amazon-ML-Hackathon/data/student_resource/outputs/matching/models"),
        REPO_ROOT / "data" / "outputs" / "matching" / "models",
        Path("/home/ec2-user/Amazon-ML-Hackathon/data/outputs/matching/models"),
    ]
    for p in candidates:
        if p.is_dir() and (p / "catboost_matcher_fold1.cbm").is_file():
            return p.resolve()
    return REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models"


MODELS_DIR = find_models_dir()
FINAL_THRESHOLD = 0.95
CHUNK_SIZE = 500_000


def build_lookup_table(df_norm: pd.DataFrame) -> Dict[str, Tuple[str, str, str, str]]:
    lookup = {}
    for eid, nn, ncl, an, cn in zip(
        df_norm["entity_id"],
        df_norm["name_norm"],
        df_norm["name_clean_legal"],
        df_norm["address_norm"],
        df_norm["country_norm"],
    ):
        lookup[str(eid)] = (
            str(nn) if pd.notna(nn) else "",
            str(ncl) if pd.notna(ncl) else "",
            str(an) if pd.notna(an) else "",
            str(cn) if pd.notna(cn) else "",
        )
    return lookup


def load_five_models() -> List[CatBoostClassifier]:
    print(f"\nLoading 5 CatBoost models from {MODELS_DIR}...")
    models = []
    for fold in range(1, 6):
        m_path = MODELS_DIR / f"catboost_matcher_fold{fold}.cbm"
        if not m_path.is_file():
            raise FileNotFoundError(f"Required model file missing: {m_path}")
        m = CatBoostClassifier()
        m.load_model(str(m_path))
        models.append(m)
        print(f"  [Fold {fold}/5] Loaded {m_path.name}")
    return models


def stream_inference_for_source(
    target_name: str,
    cand_parquet: Path,
    s1_lookup: Dict[str, Tuple[str, str, str, str]],
    target_lookup: Dict[str, Tuple[str, str, str, str]],
    models: List[CatBoostClassifier],
) -> Path:
    final_matches_file = PRED_DIR / f"test_s1_{target_name}_matches.parquet"
    if final_matches_file.is_file():
        print(f"\n[S1 -> {target_name.upper()}] Matches already generated -> {final_matches_file.name} (skipping)")
        return final_matches_file

    parts_dir = PRED_DIR / f"parts_s1_{target_name}"
    parts_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 75}")
    print(f"STREAMING INFERENCE: S1 -> {target_name.upper()}")
    print(f"Candidate file : {cand_parquet.name}")
    print(f"Ensemble Models: {len(models)}")
    print(f"Threshold      : {FINAL_THRESHOLD:.4f}")
    print(f"Chunk size     : {CHUNK_SIZE:,}")
    print(f"{'=' * 75}")

    pf = pq.ParquetFile(str(cand_parquet))
    total_candidates = pf.metadata.num_rows
    print(f"Total candidate pairs: {total_candidates:,}")

    start_time = time.time()
    processed_rows = 0
    total_matches_kept = 0
    batch_idx = 0

    existing_parts = {p.name for p in parts_dir.glob("part_*.parquet")}
    if existing_parts:
        print(f"Found {len(existing_parts):,} existing checkpoint parts. Resuming...")

    for batch in pf.iter_batches(
        batch_size=CHUNK_SIZE,
        columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"],
    ):
        part_name = f"part_{batch_idx:05d}.parquet"
        part_file = parts_dir / part_name
        n_batch = batch.num_rows

        if part_name in existing_parts:
            # Already computed
            processed_rows += n_batch
            batch_idx += 1
            continue

        t_batch = time.time()
        df_batch = batch.to_pandas()

        s1_eids = df_batch["s1_entity_id"].values
        cand_eids = df_batch["candidate_entity_id"].values
        bp_list = df_batch["blocking_passes"].values
        del df_batch

        # 1. Attribute string lookup
        s1_n = [""] * n_batch
        s1_cl = [""] * n_batch
        s1_ad = [""] * n_batch
        s1_co = [""] * n_batch
        ot_n = [""] * n_batch
        ot_cl = [""] * n_batch
        ot_ad = [""] * n_batch
        ot_co = [""] * n_batch

        for i in range(n_batch):
            s1_d = s1_lookup.get(s1_eids[i])
            if s1_d:
                s1_n[i], s1_cl[i], s1_ad[i], s1_co[i] = s1_d
            ot_d = target_lookup.get(cand_eids[i])
            if ot_d:
                ot_n[i], ot_cl[i], ot_ad[i], ot_co[i] = ot_d

        # 2. Compute 30 features
        X_batch = compute_features_batch(
            s1_names=s1_n, s1_clean=s1_cl, s1_addrs=s1_ad, s1_countries=s1_co,
            ot_names=ot_n, ot_clean=ot_cl, ot_addrs=ot_ad, ot_countries=ot_co,
            blocking_passes=bp_list,
        )
        del s1_n, s1_cl, s1_ad, s1_co, ot_n, ot_cl, ot_ad, ot_co, bp_list

        # 3. Predict with 5 models and average
        prob_sum = np.zeros(n_batch, dtype=np.float32)
        for m in models:
            prob_sum += m.predict_proba(X_batch)[:, 1].astype(np.float32)
        del X_batch

        probs = prob_sum / len(models)
        del prob_sum

        # 4. Filter matches >= threshold
        keep_mask = probs >= FINAL_THRESHOLD
        n_kept = int(np.sum(keep_mask))
        total_matches_kept += n_kept

        df_part = pd.DataFrame({
            "s1_entity_id": s1_eids[keep_mask],
            "matched_entity_id": cand_eids[keep_mask],
            "probability": probs[keep_mask],
        })
        del s1_eids, cand_eids, probs, keep_mask

        # Save part
        df_part.to_parquet(part_file, index=False, engine="pyarrow", compression="snappy")
        del df_part

        processed_rows += n_batch
        batch_idx += 1

        elapsed = time.time() - start_time
        rate = processed_rows / elapsed if elapsed > 0 else 0
        rem_rows = total_candidates - processed_rows
        eta_min = (rem_rows / rate) / 60 if rate > 0 else 0

        if batch_idx % 10 == 0 or processed_rows >= total_candidates:
            pct = (processed_rows / total_candidates) * 100
            print(
                f"  [{processed_rows:,}/{total_candidates:,}] {pct:5.1f}% | "
                f"Kept: {total_matches_kept:,} | Rate: {rate:,.0f} rows/s | ETA: {eta_min:4.1f}m | {part_name}"
            )
        gc.collect()

    # 5. Concatenate all checkpoint parts into final parquet
    print(f"\nConsolidating {batch_idx:,} parts for S1 -> {target_name.upper()}...")
    part_files = sorted(list(parts_dir.glob("part_*.parquet")))
    all_parts = []
    for pf_path in part_files:
        p_df = pd.read_parquet(pf_path)
        if len(p_df) > 0:
            all_parts.append(p_df)

    if all_parts:
        df_final = pd.concat(all_parts, ignore_index=True)
    else:
        df_final = pd.DataFrame(columns=["s1_entity_id", "matched_entity_id", "probability"])

    df_final.to_parquet(final_matches_file, index=False, engine="pyarrow", compression="snappy")
    print(f"Saved {len(df_final):,} total {target_name.upper()} match predictions to {final_matches_file.name}")

    del df_final, all_parts
    gc.collect()

    total_min = (time.time() - start_time) / 60
    print(f"Completed S1 -> {target_name.upper()} inference in {total_min:.2f} mins")
    return final_matches_file


def main():
    global FINAL_THRESHOLD
    import argparse
    parser = argparse.ArgumentParser(description="Step 3: Test Inference")
    parser.add_argument("--threshold", type=float, default=FINAL_THRESHOLD,
                        help=f"Matching threshold (default: {FINAL_THRESHOLD})")
    args, _ = parser.parse_known_args()
    
    FINAL_THRESHOLD = args.threshold
    print(f"Using operational threshold: {FINAL_THRESHOLD:.4f}")

    start_all = time.time()
    print("=" * 75)
    print(f"STEP 3: STREAMING 30-FEATURE INFERENCE (5-MODEL ENSEMBLE, THRESHOLD={FINAL_THRESHOLD})")
    print("=" * 75)

    s1_norm_path = NORM_DIR / "s1_test_normalized.parquet"
    s2_norm_path = NORM_DIR / "s2_test_normalized.parquet"
    s3_norm_path = NORM_DIR / "s3_test_normalized.parquet"

    s2_cand_path = BLOCKING_DIR / "test_s1_s2_candidates.parquet"
    s3_cand_path = BLOCKING_DIR / "test_s1_s3_candidates.parquet"

    for p in [s1_norm_path, s2_norm_path, s3_norm_path, s2_cand_path, s3_cand_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required input missing: {p}. Run 09_test_normalization.py and 10_test_blocking.py first.")

    # 1. Load models
    models = load_five_models()

    # 2. Build S1 lookup
    print(f"\nBuilding S1 lookup from {s1_norm_path.name}...")
    s1_norm = pd.read_parquet(s1_norm_path)
    s1_lookup = build_lookup_table(s1_norm)
    print(f"Loaded {len(s1_lookup):,} S1 entities.")
    del s1_norm
    gc.collect()

    # 3. Process S1 -> S2
    print(f"\nBuilding S2 lookup from {s2_norm_path.name}...")
    s2_norm = pd.read_parquet(s2_norm_path)
    s2_lookup = build_lookup_table(s2_norm)
    del s2_norm
    gc.collect()

    stream_inference_for_source("s2", s2_cand_path, s1_lookup, s2_lookup, models)
    del s2_lookup
    gc.collect()

    # 4. Process S1 -> S3
    print(f"\nBuilding S3 lookup from {s3_norm_path.name}...")
    s3_norm = pd.read_parquet(s3_norm_path)
    s3_lookup = build_lookup_table(s3_norm)
    del s3_norm
    gc.collect()

    stream_inference_for_source("s3", s3_cand_path, s1_lookup, s3_lookup, models)
    del s3_lookup, s1_lookup, models
    gc.collect()

    total_min = (time.time() - start_all) / 60
    print(f"\n{'=' * 75}")
    print(f"STEP 3 COMPLETE in {total_min:.2f} mins")
    print(f"Predictions saved to: {PRED_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
