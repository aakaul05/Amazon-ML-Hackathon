"""
scripts/11_test_inference_lightgbm.py
======================================
Step 3 of Test Pipeline: Streaming 30-Feature Computation & 5-Model LightGBM Ensemble Inference.

Pipeline Architecture:
  1. Loads all 5 trained LightGBM models (lightgbm_matcher_fold1..5.txt).
  2. Loads calibrated threshold from metrics_lightgbm.json (or CLI --threshold).
  3. Builds normalized attribute lookup dictionaries for S1, S2, S3.
  4. Streams test candidate parquet in chunks (~500,000 pairs/chunk).
  5. For each chunk:
     - Extracts attribute strings.
     - Computes exact 30 pairwise features (features.py).
     - Obtains prediction probability from each of the 5 LightGBM models.
     - Computes ensemble probability = (P1 + P2 + P3 + P4 + P5) / 5.0.
     - Retains candidate matches where ensemble probability >= THRESHOLD.
     - Releases feature matrix and probability arrays immediately from RAM.
     - Checkpoints kept matches to part_XXXX.parquet.
  6. Concatenates kept match parts into final intermediate prediction parquets:
     - data/student_resource/outputs/test/predictions/test_s1_s2_matches.parquet
     - data/student_resource/outputs/test/predictions/test_s1_s3_matches.parquet

Disk & Memory Safety:
  - Streaming chunked processing: bounds RAM usage within 16-32 GB.
  - Zero disk feature materialization: never writes 200M+ row feature matrices to disk.
  - Checkpoint resumability: skips already completed chunk parts if interrupted.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
        if p.is_dir() and (p / "lightgbm_matcher_fold1.txt").is_file():
            return p.resolve()

    candidates = [
        REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models" / "lightgbm",
        Path("/home/ec2-user/Amazon-ML-Hackathon/data/student_resource/outputs/matching/models/lightgbm"),
        REPO_ROOT / "data" / "outputs" / "matching" / "models" / "lightgbm",
        Path("/home/ec2-user/Amazon-ML-Hackathon/data/outputs/matching/models/lightgbm"),
    ]
    for p in candidates:
        if p.is_dir() and (p / "lightgbm_matcher_fold1.txt").is_file():
            return p.resolve()
    return REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models" / "lightgbm"


def load_optimal_threshold(models_dir: Path, default_threshold: float = 0.95) -> float:
    metrics_file = models_dir / "metrics_lightgbm.json"
    if metrics_file.is_file():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            th = data.get("optimal_threshold")
            if th is not None:
                print(f"Loaded calibrated threshold from {metrics_file.name}: {th:.4f}")
                return float(th)
        except Exception as e:
            print(f"Warning reading {metrics_file}: {e}")
    print(f"Using default threshold: {default_threshold:.4f}")
    return default_threshold


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


def load_five_lightgbm_models(models_dir: Path) -> List[lgb.Booster]:
    print(f"\nLoading 5 LightGBM models from {models_dir}...")
    models = []
    for fold in range(1, 6):
        m_path = models_dir / f"lightgbm_matcher_fold{fold}.txt"
        if not m_path.is_file():
            raise FileNotFoundError(f"Required model file missing: {m_path}")
        m = lgb.Booster(model_file=str(m_path))
        models.append(m)
        print(f"  [Fold {fold}/5] Loaded {m_path.name}")
    return models


def stream_inference_for_source(
    target_name: str,
    cand_parquet: Path,
    s1_lookup: Dict[str, Tuple[str, str, str, str]],
    target_lookup: Dict[str, Tuple[str, str, str, str]],
    models: List[lgb.Booster],
    threshold: float,
    chunk_size: int = 500_000,
) -> Path:
    final_matches_file = PRED_DIR / f"test_s1_{target_name}_matches.parquet"
    if final_matches_file.is_file():
        print(f"\n[S1 -> {target_name.upper()}] Matches already generated -> {final_matches_file.name} (skipping)")
        return final_matches_file

    parts_dir = PRED_DIR / f"parts_s1_{target_name}_lgb"
    parts_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 75}")
    print(f"STREAMING INFERENCE (LIGHTGBM): S1 -> {target_name.upper()}")
    print(f"Candidate file : {cand_parquet.name}")
    print(f"Ensemble Models: {len(models)}")
    print(f"Threshold      : {threshold:.4f}")
    print(f"Chunk size     : {chunk_size:,}")
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
        batch_size=chunk_size,
        columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"],
    ):
        part_name = f"part_{batch_idx:05d}.parquet"
        part_file = parts_dir / part_name
        n_batch = batch.num_rows

        if part_name in existing_parts:
            processed_rows += n_batch
            batch_idx += 1
            continue

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

        # 3. Predict with 5 LightGBM models and average
        prob_sum = np.zeros(n_batch, dtype=np.float32)
        for m in models:
            prob_sum += m.predict(X_batch).astype(np.float32)
        del X_batch

        probs = prob_sum / len(models)
        del prob_sum

        # 4. Filter matches >= threshold
        keep_mask = probs >= threshold
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

    # Concatenate all checkpoint parts into final parquet
    print(f"\nConcatenating {batch_idx:,} parts into {final_matches_file.name}...")
    part_files = sorted(list(parts_dir.glob("part_*.parquet")))
    if part_files:
        all_parts = [pd.read_parquet(pf) for pf in part_files]
        df_matches = pd.concat(all_parts, ignore_index=True)
        df_matches.to_parquet(final_matches_file, index=False, engine="pyarrow", compression="snappy")
        del all_parts, df_matches
        gc.collect()
    else:
        pd.DataFrame(columns=["s1_entity_id", "matched_entity_id", "probability"]).to_parquet(
            final_matches_file, index=False
        )

    print(f"Saved: {final_matches_file} ({total_matches_kept:,} matches kept)")
    return final_matches_file


def main():
    parser = argparse.ArgumentParser(description="Streaming LightGBM Test Inference")
    parser.add_argument("--threshold", type=float, default=None, help="Inference threshold override")
    parser.add_argument("--chunk-size", type=int, default=500_000, help="Candidate chunk size")
    args = parser.parse_args()

    models_dir = find_models_dir()
    threshold = args.threshold if args.threshold is not None else load_optimal_threshold(models_dir)

    print("=" * 80)
    print("STEP 3: TEST CANDIDATE STREAMING INFERENCE WITH 5-MODEL LIGHTGBM ENSEMBLE")
    print("=" * 80)

    # 1. Load models
    models = load_five_lightgbm_models(models_dir)

    def find_norm_file(norm_dir: Path, src: str) -> Path:
        for name in [f"{src}_test_normalized.parquet", f"test_{src}_normalized.parquet"]:
            p = norm_dir / name
            if p.is_file():
                return p
        raise FileNotFoundError(f"Normalized file for {src} not found in {norm_dir}")

    # 2. Load normalized tables
    print("\nLoading normalized entity tables...")
    t0 = time.time()
    s1_norm_path = find_norm_file(NORM_DIR, "s1")
    s2_norm_path = find_norm_file(NORM_DIR, "s2")
    s3_norm_path = find_norm_file(NORM_DIR, "s3")

    df_s1 = pd.read_parquet(s1_norm_path)
    df_s2 = pd.read_parquet(s2_norm_path)
    df_s3 = pd.read_parquet(s3_norm_path)
    print(f"Loaded tables in {time.time() - t0:.2f}s ({s1_norm_path.name}, {s2_norm_path.name}, {s3_norm_path.name})")

    print("\nBuilding memory-efficient attribute lookup dictionaries...")
    t0 = time.time()
    s1_lookup = build_lookup_table(df_s1)
    del df_s1
    gc.collect()

    s2_lookup = build_lookup_table(df_s2)
    del df_s2
    gc.collect()

    s3_lookup = build_lookup_table(df_s3)
    del df_s3
    gc.collect()
    print(f"Lookups built in {time.time() - t0:.2f}s (S1: {len(s1_lookup):,}, S2: {len(s2_lookup):,}, S3: {len(s3_lookup):,})")

    # 3. Stream inference for S2
    s2_cands = BLOCKING_DIR / "test_s1_s2_candidates.parquet"
    stream_inference_for_source("s2", s2_cands, s1_lookup, s2_lookup, models, threshold, chunk_size=args.chunk_size)
    del s2_lookup
    gc.collect()

    # 4. Stream inference for S3
    s3_cands = BLOCKING_DIR / "test_s1_s3_candidates.parquet"
    stream_inference_for_source("s3", s3_cands, s1_lookup, s3_lookup, models, threshold, chunk_size=args.chunk_size)
    del s1_lookup, s3_lookup
    gc.collect()

    print("\n" + "=" * 80)
    print("STEP 3 LIGHTGBM TEST INFERENCE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
