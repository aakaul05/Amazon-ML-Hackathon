"""
scripts/create_validation_benchmark.py
======================================
Creates a deterministic 10,000 S1 entity validation benchmark (Phase 4).
Uses RANDOM_SEED = 42 and maintains representative country distribution.
Extracts all corresponding ground-truth pairs for S2 and S3.
"""

from pathlib import Path
import os
import sys
import json
import pandas as pd
import numpy as np
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = REPO_ROOT / "data" / "outputs" / "normalized_cache"
GT_PATH = REPO_ROOT / "data" / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"
OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SAMPLE_SIZE = 10_000

def main():
    print("=" * 70)
    print("PHASE 4: CREATING DETERMINISTIC 10,000 S1 VALIDATION BENCHMARK")
    print("=" * 70)

    # 1. Load S1 entities
    s1_path = NORMALIZED_DIR / "s1_normalized.parquet"
    print(f"Loading S1 from {s1_path}...")
    s1_df = pd.read_parquet(
        s1_path,
        columns=["entity_id", "business_name", "business_address", "country", "name_norm", "name_clean_legal", "address_norm", "country_norm"]
    )
    print(f"Total S1 entities: {len(s1_df):,}")

    # 2. Stratified sample by country_norm
    country_counts = s1_df["country_norm"].value_counts()
    print("\nS1 Country Distribution:")
    for c, cnt in country_counts.items():
        print(f"  {c}: {cnt:,} ({cnt / len(s1_df) * 100:.2f}%)")

    # Sample proportionally
    sampled_dfs = []
    total_sampled = 0
    countries = list(country_counts.index)
    
    for i, country in enumerate(countries):
        country_sub = s1_df[s1_df["country_norm"] == country]
        if i == len(countries) - 1:
            n_target = SAMPLE_SIZE - total_sampled
        else:
            n_target = int(round(SAMPLE_SIZE * (len(country_sub) / len(s1_df))))
        
        sample = country_sub.sample(n=n_target, random_state=RANDOM_SEED)
        sampled_dfs.append(sample)
        total_sampled += len(sample)
        print(f"Sampled {len(sample):,} entities for country: {country}")

    val_s1 = pd.concat(sampled_dfs, ignore_index=True).sort_values("entity_id").reset_index(drop=True)
    print(f"\nFinal validation set size: {len(val_s1):,} S1 entities")

    # 3. Load Ground Truth for validation entities
    val_s1_set = set(val_s1["entity_id"])
    gt_s2 = defaultdict(set)
    gt_s3 = defaultdict(set)
    total_gt_s2 = 0
    total_gt_s3 = 0
    singletons = 0

    print(f"\nExtracting ground-truth pairs from {GT_PATH}...")
    with open(GT_PATH, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            s1_id = parts[0].strip()
            if s1_id not in val_s1_set:
                continue
            
            matches = parts[1].strip()
            if not matches:
                singletons += 1
                continue
            
            has_match = False
            for m in matches.split(","):
                m = m.strip()
                if m.startswith("S2-"):
                    gt_s2[s1_id].add(m)
                    total_gt_s2 += 1
                    has_match = True
                elif m.startswith("S3-"):
                    gt_s3[s1_id].add(m)
                    total_gt_s3 += 1
                    has_match = True
            
            if not has_match:
                singletons += 1

    # Entities with zero GT matches
    matched_s1 = set(gt_s2.keys()) | set(gt_s3.keys())
    actual_singletons = len(val_s1_set - matched_s1)

    print("\nGround Truth Summary on 10k Validation Set:")
    print(f"  S1 entities with matches : {len(matched_s1):,} ({len(matched_s1)/SAMPLE_SIZE*100:.2f}%)")
    print(f"  S1 singletons (no matches): {actual_singletons:,} ({actual_singletons/SAMPLE_SIZE*100:.2f}%)")
    print(f"  Total true S2 pairs       : {total_gt_s2:,} (across {len(gt_s2):,} S1 entities)")
    print(f"  Total true S3 pairs       : {total_gt_s3:,} (across {len(gt_s3):,} S1 entities)")

    # 4. Save validation files
    val_s1_path = OUTPUT_DIR / "val_s1_10k.parquet"
    val_s1.to_parquet(val_s1_path, index=False)
    print(f"\nSaved validation S1 entities: {val_s1_path}")

    # Save GT as json dictionary
    gt_export = {
        "s2": {k: sorted(list(v)) for k, v in gt_s2.items()},
        "s3": {k: sorted(list(v)) for k, v in gt_s3.items()},
        "stats": {
            "sample_size": SAMPLE_SIZE,
            "random_seed": RANDOM_SEED,
            "total_s2_pairs": total_gt_s2,
            "total_s3_pairs": total_gt_s3,
            "s1_with_s2_match": len(gt_s2),
            "s1_with_s3_match": len(gt_s3),
            "s1_singletons": actual_singletons,
            "country_distribution": {c: int((val_s1["country_norm"] == c).sum()) for c in countries}
        }
    }
    gt_json_path = OUTPUT_DIR / "val_gt_10k.json"
    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(gt_export, f, indent=2)
    print(f"Saved ground truth: {gt_json_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
