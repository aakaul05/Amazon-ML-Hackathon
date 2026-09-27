"""
scripts/10_test_blocking.py
===========================
Step 2 of Test Pipeline: Test Candidate Generation (Task 7 10-Pass Blocking).

Runs the verified Task 7 10-pass blocking pipeline on the test dataset:
  1. Exact name_norm
  2. Exact clean legal name
  3. Rare tokens: max_df=800, max_cand_per_s1=80
  4. Address component tokens
  5. TF-IDF char 3-gram: min_sim=0.70, top_k=10, sample_limit=500000
  6. TF-IDF char 3-gram relaxed: min_sim=0.45, top_k=20, sample_limit=500000
  7. Sorted tokens: max_key_df=500, max_cand_per_s1=50
  8. Name prefix 6: max_key_df=200, max_cand_per_s1=30
  9. Country + Name compound: max_key_df=500, max_cand_per_s1=60, min_token_len=4
  10. Phonetic (Double Metaphone): max_key_df=300, max_cand_per_s1=40

Combines all passes using bitmask encoding without requiring ground truth.
Guarantees deduplication of (s1_entity_id, candidate_entity_id) pairs.

Outputs saved to:
  data/student_resource/outputs/test/blocking/
    - test_s1_s2_candidates.parquet
    - test_s1_s3_candidates.parquet

Supports resume: skips a target source if its candidates parquet already exists.
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.blocking.candidate_generator import (
    block_exact_field,
    block_rare_tokens,
    block_address_tokens,
    block_tfidf_char_ngram,
    block_sorted_tokens,
    block_name_prefix,
    block_country_name_token,
    block_phonetic,
    BLOCK_BITMASK,
)

NORM_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "test" / "normalized"
BLOCKING_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "test" / "blocking"
BLOCKING_DIR.mkdir(parents=True, exist_ok=True)


def combine_test_blocks(block_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Unions candidates across blocking passes and encodes the bitmask
    into pipe-separated 'blocking_passes' string without requiring ground truth.
    Guarantees zero duplicate (s1_entity_id, candidate_entity_id) pairs.
    """
    valid_dfs = [df for df in block_dfs if df is not None and not df.empty]
    if not valid_dfs:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])

    # Ensure relaxed char ngram bitmask is supported for both 0.45 and 0.50
    active_bitmask = dict(BLOCK_BITMASK)
    if "char_ngram_0.45" not in active_bitmask:
        active_bitmask["char_ngram_0.45"] = 32
    if "char_ngram_0.50" not in active_bitmask:
        active_bitmask["char_ngram_0.50"] = 32

    present_blocks = {}
    for df in valid_dfs:
        bname = df["block"].iloc[0]
        if bname in active_bitmask:
            present_blocks[bname] = active_bitmask[bname]

    max_bits = max(present_blocks.values()) * 2 if present_blocks else 32
    bitmask_to_str = {}
    for i in range(1, max_bits):
        passes = []
        for bname, bit_val in sorted(present_blocks.items(), key=lambda x: x[1]):
            if i & bit_val:
                passes.append(bname)
        if passes:
            bitmask_to_str[i] = "|".join(passes)

    # Assign bitmasks and concatenate
    dfs_with_mask = []
    for df in valid_dfs:
        bname = df["block"].iloc[0]
        bit_val = active_bitmask.get(bname, 0)
        if bit_val == 0:
            continue
        sub = df[["s1_entity_id", "candidate_entity_id"]].drop_duplicates().copy()
        sub["mask"] = np.uint16(bit_val)
        dfs_with_mask.append(sub)

    concat_df = pd.concat(dfs_with_mask, ignore_index=True)
    del dfs_with_mask
    gc.collect()

    # Groupby sum across distinct powers of 2 (equivalent to bitwise OR)
    print("  Aggregating bitmasks and deduplicating candidate pairs...")
    cand_df = concat_df.groupby(
        ["s1_entity_id", "candidate_entity_id"], as_index=False
    )["mask"].sum()
    del concat_df
    gc.collect()

    cand_df["blocking_passes"] = cand_df["mask"].map(bitmask_to_str).fillna("")
    cand_df.drop(columns=["mask"], inplace=True)
    return cand_df


def run_all_task7_blocks(s1: pd.DataFrame, other: pd.DataFrame, target_name: str) -> List[pd.DataFrame]:
    """Runs all 10 verified Task 7 blocking passes."""
    print(f"\n{'#' * 75}")
    print(f"# BLOCKING: S1 -> {target_name.upper()}")
    print(f"# S1: {len(s1):,} rows  |  {target_name.upper()}: {len(other):,} rows")
    print(f"{'#' * 75}")

    blocks = []

    def _execute(name: str, fn):
        t0 = time.time()
        print(f"\n--- Starting {name} ---", flush=True)
        try:
            res = fn()
            count = len(res) if res is not None else 0
            print(f"--- Finished {name}: {count:,} pairs ({time.time() - t0:.1f}s) ---", flush=True)
            gc.collect()
            return res
        except Exception as e:
            print(f"ERROR in {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "block"])

    # Pass 1: Exact name_norm
    blocks.append(_execute(
        "Pass 1: exact_name_norm",
        lambda: block_exact_field(s1, other, "name_norm", "exact_name_norm")
    ))

    # Pass 2: Exact clean legal name
    blocks.append(_execute(
        "Pass 2: exact_clean_legal",
        lambda: block_exact_field(s1, other, "name_clean_legal", "exact_clean_legal")
    ))

    # Pass 3: Rare tokens (max_df=800, max_cand_per_s1=80)
    blocks.append(_execute(
        "Pass 3: rare_token (DF<=800)",
        lambda: block_rare_tokens(s1, other, max_df=800, max_cand_per_s1=80)
    ))

    # Pass 4: Address component tokens
    blocks.append(_execute(
        "Pass 4: address_component",
        lambda: block_address_tokens(s1, other)
    ))

    # Pass 5: TF-IDF char 3-gram (0.70)
    blocks.append(_execute(
        "Pass 5: char_ngram_0.70 (sim>=0.70, top_k=10)",
        lambda: block_tfidf_char_ngram(s1, other, min_sim=0.70, top_k=10, sample_limit=500000)
    ))

    # Pass 6: TF-IDF char 3-gram relaxed (0.45)
    blocks.append(_execute(
        "Pass 6: char_ngram_0.45 (sim>=0.45, top_k=20)",
        lambda: block_tfidf_char_ngram(s1, other, min_sim=0.45, top_k=20, sample_limit=500000)
    ))

    # Pass 7: Sorted-token blocking (max_key_df=500, max_cand_per_s1=50)
    blocks.append(_execute(
        "Pass 7: sorted_token",
        lambda: block_sorted_tokens(s1, other, max_key_df=500, max_cand_per_s1=50)
    ))

    # Pass 8: Name prefix blocking 6 chars (max_key_df=200, max_cand_per_s1=30)
    blocks.append(_execute(
        "Pass 8: name_prefix_6",
        lambda: block_name_prefix(s1, other, prefix_len=6, max_key_df=200, max_cand_per_s1=30)
    ))

    # Pass 9: Country + Name compound (max_key_df=500, max_cand_per_s1=60, min_len=4)
    blocks.append(_execute(
        "Pass 9: country_name_token",
        lambda: block_country_name_token(s1, other, max_key_df=500, max_cand_per_s1=60, min_token_len=4)
    ))

    # Pass 10: Phonetic blocking (max_key_df=300, max_cand_per_s1=40)
    blocks.append(_execute(
        "Pass 10: phonetic",
        lambda: block_phonetic(s1, other, max_key_df=300, max_cand_per_s1=40)
    ))

    return blocks


def process_target_source(s1_df: pd.DataFrame, target_parquet: Path, target_name: str) -> Path:
    out_file = BLOCKING_DIR / f"test_s1_{target_name}_candidates.parquet"
    if out_file.is_file():
        print(f"\n[S1 -> {target_name.upper()}] Candidates already generated -> {out_file.name} (skipping)")
        return out_file

    t0 = time.time()
    print(f"\nLoading normalized {target_name.upper()} from {target_parquet.name}...")
    other_df = pd.read_parquet(target_parquet)
    print(f"Loaded {len(other_df):,} rows.")

    # Run blocking
    blocks = run_all_task7_blocks(s1_df, other_df, target_name)
    del other_df
    gc.collect()

    # Combine blocks
    print(f"\nCombining and deduplicating S1 -> {target_name.upper()} candidates...")
    t_comb = time.time()
    cand_df = combine_test_blocks(blocks)
    del blocks
    gc.collect()

    print(f"Combined {len(cand_df):,} unique candidate pairs in {time.time() - t_comb:.1f}s")

    # Save to parquet
    print(f"Saving to {out_file.name}...")
    cand_df.to_parquet(out_file, index=False, engine="pyarrow", compression="snappy")
    del cand_df
    gc.collect()

    elapsed = (time.time() - t0) / 60
    print(f"Completed S1 -> {target_name.upper()} candidate generation in {elapsed:.2f} mins")
    return out_file


def main():
    start_all = time.time()
    print("=" * 75)
    print("STEP 2: TEST CANDIDATE GENERATION (TASK 7 10-PASS BLOCKING)")
    print("=" * 75)

    s1_norm_path = NORM_DIR / "s1_test_normalized.parquet"
    s2_norm_path = NORM_DIR / "s2_test_normalized.parquet"
    s3_norm_path = NORM_DIR / "s3_test_normalized.parquet"

    for p in [s1_norm_path, s2_norm_path, s3_norm_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Normalized cache file missing: {p}. Run 09_test_normalization.py first.")

    print(f"Loading normalized S1 from {s1_norm_path.name}...")
    s1_df = pd.read_parquet(s1_norm_path)
    print(f"Loaded {len(s1_df):,} S1 rows.")

    # Process S1 -> S2
    process_target_source(s1_df, s2_norm_path, "s2")

    # Process S1 -> S3
    process_target_source(s1_df, s3_norm_path, "s3")

    del s1_df
    gc.collect()

    total_min = (time.time() - start_all) / 60
    print(f"\n{'=' * 75}")
    print(f"STEP 2 COMPLETE in {total_min:.2f} mins")
    print(f"Outputs saved to: {BLOCKING_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
