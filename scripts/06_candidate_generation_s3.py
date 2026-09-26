"""
scripts/06_candidate_generation_s3.py
======================================
Business Entity Resolution — Task 6 (S3 Only): Candidate Generation / Blocking

Standalone, memory-optimized blocking pipeline for Source 3 ONLY.
Run this AFTER script 04 (normalization) has created the parquet caches
and after the main 06_candidate_generation.py has saved s1_s2_candidates.

Key memory optimisation vs the combined script:
  - Loads ONLY S1 + S3 parquet caches (never touches S2)
  - Frees each block result immediately after evaluation

Multi-Pass Blocking System:
 1. Exact Normalized Name (`name_norm`)
 2. Exact Clean Legal Name (`name_clean_legal`)
 3. Informative / Rare Name Tokens (DF thresholded)
 4. Address Component & Alphanumeric Blocking (`address_norm`)
 5. TF-IDF Character N-Gram Sparse Nearest Neighbor Retrieval

Outputs:
 - data/student_resource/outputs/blocking/s1_s3_candidates.parquet
 - data/student_resource/outputs/blocking/blocking_statistics_s3.csv
"""

import gc
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd

# Reconfigure stdout for UTF-8 line buffering on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

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

OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Normalized parquet cache directory (created by script 04)
CACHE_DIR = REPO_ROOT / "data" / "outputs" / "normalized_cache"


# ============================================================
# Path Discovery
# ============================================================

def find_dataset_dir():
    """Locate the training dataset directory."""
    for env_var in ("BER_DATA_DIR", "DATA_DIR"):
        env_val = os.environ.get(env_var)
        if env_val:
            p = Path(env_val)
            if p.is_dir():
                return p.resolve()

    candidates = [
        REPO_ROOT / "data" / "student_resource" / "dataset" / "train",
        REPO_ROOT / "data" / "dataset" / "train",
        REPO_ROOT / "data" / "train",
        REPO_ROOT / "dataset" / "student_resource" / "dataset" / "train",
        REPO_ROOT / "dataset" / "train",
    ]

    for p in candidates:
        if p.is_dir() and (p / "train_ground_truth.tsv").is_file():
            return p.resolve()

    raise FileNotFoundError("Training dataset directory not found. Set BER_DATA_DIR.")


TRAIN_DIR = find_dataset_dir()


# ============================================================
# Ground Truth (S3 only)
# ============================================================

def parse_ground_truth_s3():
    """Stream-parse ground truth, keeping only S3 matches."""
    gt_s3 = defaultdict(set)
    total_s3 = 0

    path = TRAIN_DIR / "train_ground_truth.tsv"

    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue

            s1_id = parts[0].strip()
            matches = parts[1].strip()
            if not s1_id or not matches:
                continue

            for m in matches.split(","):
                m = m.strip()
                if m.startswith("S3-"):
                    gt_s3[s1_id].add(m)
                    total_s3 += 1

    return gt_s3, total_s3


# ============================================================
# Memory-Efficient Data Loading (S1 + S3 only)
# ============================================================

def load_s1_s3(columns):
    """
    Load ONLY S1 and S3 normalized parquet caches.
    Unlike load_normalized_or_compute(), this never loads S2 into memory.
    Falls back to load_normalized_or_compute() if caches are missing.
    """
    s1_cache = CACHE_DIR / "s1_normalized.parquet"
    s3_cache = CACHE_DIR / "s3_normalized.parquet"

    if s1_cache.exists() and s3_cache.exists():
        print(f"Loading S1 + S3 parquet caches directly (skipping S2)...")
        s1 = pd.read_parquet(s1_cache, columns=columns)
        s3 = pd.read_parquet(s3_cache, columns=columns)
        return s1, s3

    # Fallback: caches don't exist yet — use the full loader and discard S2
    print("WARNING: Parquet caches not found, falling back to full loader (loads S2 too).")
    from business_entity_resolution.preprocessing.normalization import (
        load_normalized_or_compute,
    )
    s1, s2, s3 = load_normalized_or_compute(TRAIN_DIR, REPO_ROOT, columns=columns)
    del s2
    gc.collect()
    return s1, s3


# ============================================================
# Block Runner
# ============================================================

def run_pass(name, func):
    """Run a blocking pass with timing and error handling."""
    print("\n" + "=" * 70, flush=True)
    print(f"STARTING {name}", flush=True)
    print("=" * 70, flush=True)

    start = time.time()
    try:
        result = func()
        print(f"FINISHED {name} in {time.time() - start:.1f}s", flush=True)
        gc.collect()
        return result
    except Exception as e:
        print(f"\nERROR IN {name}: {e}", flush=True)
        raise


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("=" * 70)
    print("TASK 6 — SOURCE 3 ONLY (MEMORY-OPTIMIZED)")
    print("=" * 70)
    print(f"Dataset : {TRAIN_DIR}")
    print(f"Cache   : {CACHE_DIR}")
    print(f"Output  : {OUTPUT_DIR}")

    # ── Ground Truth ──────────────────────────────────────────
    gt_s3, len_gt_s3 = parse_ground_truth_s3()
    print(f"\nGround truth S3 pairs: {len_gt_s3:,}")

    # ── Load S1 + S3 ONLY ────────────────────────────────────
    cols = ["entity_id", "name_norm", "name_clean_legal", "address_norm"]
    s1, s3 = load_s1_s3(cols)

    print(f"S1 rows: {len(s1):,}")
    print(f"S3 rows: {len(s3):,}")

    # ── Block 1: Exact name_norm ─────────────────────────────
    b1 = run_pass(
        "Block 1 — Exact name_norm",
        lambda: block_exact_field(s1, s3, "name_norm", "exact_name_norm"),
    )

    # ── Block 2: Exact clean legal name ──────────────────────
    b2 = run_pass(
        "Block 2 — Exact clean legal name",
        lambda: block_exact_field(s1, s3, "name_clean_legal", "exact_clean_legal"),
    )

    # ── Block 3: Rare tokens ─────────────────────────────────
    b3 = run_pass(
        "Block 3 — Rare tokens",
        lambda: block_rare_tokens(s1, s3, max_df=300, max_cand_per_s1=50),
    )

    # ── Block 4: Address tokens ──────────────────────────────
    b4 = run_pass(
        "Block 4 — Address tokens",
        lambda: block_address_tokens(s1, s3),
    )

    # ── Block 5: TF-IDF char n-gram ─────────────────────────
    b5 = run_pass(
        "Block 5 — TF-IDF char n-gram",
        lambda: block_tfidf_char_ngram(
            s1, s3, min_sim=0.70, top_k=10, sample_limit=300000,
        ),
    )

    # ── Combine + Evaluate ───────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATING SOURCE 3")
    print("=" * 70)

    candidates, stats = combine_blocks_and_evaluate(
        s1, s3, gt_s3, len_gt_s3,
        [b1, b2, b3, b4, b5],
        "Source 3",
    )

    # Free block results immediately
    del b1, b2, b3, b4, b5, s3
    gc.collect()

    # ── Save Candidates ──────────────────────────────────────
    output = OUTPUT_DIR / "s1_s3_candidates.parquet"
    candidates.to_parquet(output, index=False)
    print(f"\nSaved: {output}")
    print(f"Candidate pairs: {len(candidates):,}")

    # ── Save Statistics ──────────────────────────────────────
    stats_output = OUTPUT_DIR / "blocking_statistics_s3.csv"
    pd.DataFrame([stats]).to_csv(stats_output, index=False)
    print(f"Saved: {stats_output}")

    # Final cleanup
    del candidates, s1
    gc.collect()

    print("\n" + "=" * 70)
    print("SOURCE 3 BLOCKING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
