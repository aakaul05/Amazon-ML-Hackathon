"""
scripts/06_candidate_generation.py
===================================
Business Entity Resolution — Task 6: Candidate Generation / Blocking

Multi-Pass Blocking System:
1. Block 1: Exact Normalized Name (`name_norm`)
2. Block 2: Exact Clean Legal Name (`name_clean_legal`)
3. Block 3: Informative / Rare Name Tokens (DF thresholded)
4. Block 4: Address Component & Alphanumeric Blocking (`address_norm`)
5. Block 5: Fast TF-IDF Character N-Gram Sparse Nearest Neighbor Retrieval

Outputs:
- data/student_resource/outputs/blocking/s1_s2_candidates.parquet
- data/student_resource/outputs/blocking/s1_s3_candidates.parquet
- data/student_resource/outputs/blocking/blocking_statistics.csv
"""

import gc
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from sparse_dot_topn import awesome_cossim_topn

# Reconfigure stdout for UTF-8 line buffering on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Determine repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.blocking import (
    block_exact_field,
    block_rare_tokens,
    block_address_tokens,
    block_tfidf_char_ngram,
    combine_blocks_and_evaluate,
)
from business_entity_resolution.preprocessing.normalization import load_normalized_or_compute

OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Path Discovery & Dataset Loading
# ============================================================

def find_dataset_dir(repo_root: Path) -> Path:
    search_paths = []
    for env_var in ("BER_DATA_DIR", "DATA_DIR"):
        env_val = os.environ.get(env_var)
        if env_val:
            search_paths.append(Path(env_val))

    search_paths.extend([
        repo_root / "data" / "student_resource" / "dataset" / "train",
        repo_root / "data" / "dataset" / "train",
        repo_root / "data" / "train",
        repo_root / "dataset" / "train",
    ])

    required_files = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv",
    ]

    for candidate in search_paths:
        if candidate.is_dir():
            missing = [f for f in required_files if not (candidate / f).is_file()]
            if not missing:
                return candidate.resolve()

    raise FileNotFoundError("Training dataset files not found. Check paths.")


TRAIN_DIR = find_dataset_dir(REPO_ROOT)

print("=" * 70)
print("TASK 6: CANDIDATE GENERATION / BLOCKING")
print("=" * 70)
print(f"Dataset Dir: {TRAIN_DIR}")
print(f"Output Dir : {OUTPUT_DIR}\n")


def load_and_normalize_all():
    print("Loading normalized datasets...")
    s1, s2, s3 = load_normalized_or_compute(TRAIN_DIR, REPO_ROOT)

    gt = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        usecols=["source1_entity_id", "matched_entity_ids"],
        dtype=str,
    )

    # Ground truth parsing
    print("Parsing Ground Truth...")
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    gt["matched_id"] = gt["matched_entity_ids"].str.split(",")
    exploded = gt.explode("matched_id", ignore_index=True)
    exploded["matched_id"] = exploded["matched_id"].str.strip()
    exploded = exploded[exploded["matched_id"] != ""].copy()
    exploded = exploded.rename(columns={"source1_entity_id": "s1_id", "matched_id": "other_id"})

    gt_s2 = exploded[exploded["other_id"].str.startswith("S2-")].copy()
    gt_s3 = exploded[exploded["other_id"].str.startswith("S3-")].copy()

    gt_dict_s2 = gt_s2.groupby("s1_id")["other_id"].apply(set).to_dict()
    gt_dict_s3 = gt_s3.groupby("s1_id")["other_id"].apply(set).to_dict()

    return s1, s2, s3, gt_dict_s2, gt_dict_s3, len(gt_s2), len(gt_s3)

    stats = {
        "source": source_name,
        "total_candidates": total_candidates,
        "total_tp_recovered": total_tp,
        "total_true_pairs": total_true_pairs,
        "candidate_recall": union_recall,
        "complete_s1_coverage_pct": full_cov_pct,
        "zero_s1_coverage_pct": zero_cov_pct,
        "reduction_ratio": reduction_ratio,
    }

    return cand_df, stats


# ============================================================
# Main Execution Pipeline
# ============================================================

def main():
    s1, s2, s3, gt_s2, gt_s3, len_gt_s2, len_gt_s3 = load_and_normalize_all()

    # --- SOURCE 2 BLOCKING ---
    print("\n" + "=" * 70)
    print("GENERATING BLOCKING PASSES FOR SOURCE 2")
    print("=" * 70)
    b1_s2 = block_exact_field(s1, s2, "name_norm", "exact_name_norm")
    b2_s2 = block_exact_field(s1, s2, "name_clean_legal", "exact_clean_legal")
    b3_s2 = block_rare_tokens(s1, s2, max_df=500, max_cand_per_s1=50)
    b4_s2 = block_address_tokens(s1, s2)
    b5_s2 = block_tfidf_char_ngram(s1, s2, min_sim=0.70, top_k=10, sample_limit=300000)

    cand_s2_df, stats_s2 = combine_blocks_and_evaluate(
        s1, s2, gt_s2, len_gt_s2, [b1_s2, b2_s2, b3_s2, b4_s2, b5_s2], "Source 2"
    )

    # Save S2 Parquet
    out_s2_path = OUTPUT_DIR / "s1_s2_candidates.parquet"
    cand_s2_df.to_parquet(out_s2_path, index=False)
    print(f"Saved: {out_s2_path}")

    del b1_s2, b2_s2, b3_s2, b4_s2, b5_s2, cand_s2_df
    gc.collect()

    # --- SOURCE 3 BLOCKING ---
    print("\n" + "=" * 70)
    print("GENERATING BLOCKING PASSES FOR SOURCE 3")
    print("=" * 70)
    b1_s3 = block_exact_field(s1, s3, "name_norm", "exact_name_norm")
    b2_s3 = block_exact_field(s1, s3, "name_clean_legal", "exact_clean_legal")
    b3_s3 = block_rare_tokens(s1, s3, max_df=500, max_cand_per_s1=50)
    b4_s3 = block_address_tokens(s1, s3)
    b5_s3 = block_tfidf_char_ngram(s1, s3, min_sim=0.70, top_k=10, sample_limit=300000)

    cand_s3_df, stats_s3 = combine_blocks_and_evaluate(
        s1, s3, gt_s3, len_gt_s3, [b1_s3, b2_s3, b3_s3, b4_s3, b5_s3], "Source 3"
    )

    # Save S3 Parquet
    out_s3_path = OUTPUT_DIR / "s1_s3_candidates.parquet"
    cand_s3_df.to_parquet(out_s3_path, index=False)
    print(f"Saved: {out_s3_path}")

    del b1_s3, b2_s3, b3_s3, b4_s3, b5_s3, cand_s3_df
    gc.collect()

    # Save Blocking Statistics CSV
    stats_df = pd.DataFrame([stats_s2, stats_s3])
    out_stats_path = OUTPUT_DIR / "blocking_statistics.csv"
    stats_df.to_csv(out_stats_path, index=False)
    print(f"Saved: {out_stats_path}")

    print("\n" + "=" * 70)
    print("TASK 6 CANDIDATE GENERATION / BLOCKING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
