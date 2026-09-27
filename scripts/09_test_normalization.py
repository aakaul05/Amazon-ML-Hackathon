"""
scripts/09_test_normalization.py
================================
Step 1 of Test Pipeline: Test Data Normalization.

Normalizes the 3 test sources:
  - test_source1.tsv (1,732,544 rows)
  - test_source2.tsv (4,887,273 rows)
  - test_source3.tsv (5,082,316 rows)

Using the exact same normalization logic as training:
  - basic_clean_string (NFKD accent decomposition, lowercasing, punctuation -> space, whitespace collapse)
  - strip_legal_suffix (recursive stripping of corporate designations from end of name)
  - normalize_series_fast (cache across unique strings to optimize memory and speed)
  - Open-set country normalization (works seamlessly on US, India, France)

Outputs saved to:
  data/student_resource/outputs/test/normalized/
    - s1_test_normalized.parquet
    - s2_test_normalized.parquet
    - s3_test_normalized.parquet

Supports resume: skips sources that have already been normalized.
"""

import gc
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.preprocessing.normalization import (
    normalize_series_fast,
)


def find_test_dir() -> Path:
    for env_var in ("BER_TEST_DIR", "BER_DATA_DIR", "DATA_DIR"):
        env_val = os.environ.get(env_var)
        if env_val:
            p = Path(env_val)
            if p.is_dir() and (p / "test_source1.tsv").is_file():
                return p.resolve()
            if (p / "test" / "test_source1.tsv").is_file():
                return (p / "test").resolve()

    candidates = [
        REPO_ROOT / "dataset" / "student_resource" / "dataset" / "test",
        Path("/home/ec2-user/Amazon-ML-Hackathon/dataset/student_resource/dataset/test"),
        REPO_ROOT / "data" / "student_resource" / "dataset" / "test",
        Path.home() / "Amazon-ML-Hackathon" / "dataset" / "student_resource" / "dataset" / "test",
    ]
    for p in candidates:
        if p.is_dir() and (p / "test_source1.tsv").is_file():
            return p.resolve()

    raise FileNotFoundError("Test dataset directory not found. Expected test_source1.tsv.")


TEST_DIR = find_test_dir()
OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "test" / "normalized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_source_file(tsv_name: str, parquet_name: str, source_label: str) -> Path:
    target_parquet = OUTPUT_DIR / parquet_name
    if target_parquet.is_file():
        print(f"[{source_label}] Already normalized -> {target_parquet.name} (skipping)")
        return target_parquet

    tsv_path = TEST_DIR / tsv_name
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Source file not found: {tsv_path}")

    print(f"\n{'=' * 75}")
    print(f"NORMALIZING {source_label}: {tsv_path.name}")
    print(f"{'=' * 75}")
    t0 = time.time()

    cols = ["entity_id", "business_name", "business_address", "country"]
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, usecols=cols)
    df = df.fillna("")
    n_rows = len(df)
    print(f"Loaded {n_rows:,} rows in {time.time() - t0:.1f}s")

    # Name normalization
    t_sub = time.time()
    df["name_norm"], df["name_clean_legal"] = normalize_series_fast(df["business_name"], is_name=True)
    print(f"  Name normalization: {time.time() - t_sub:.1f}s")

    # Address normalization
    t_sub = time.time()
    df["address_norm"], _ = normalize_series_fast(df["business_address"], is_name=False)
    print(f"  Address normalization: {time.time() - t_sub:.1f}s")

    # Country normalization (open-set: India, US, France, etc.)
    t_sub = time.time()
    df["country_norm"], _ = normalize_series_fast(df["country"], is_name=False)
    print(f"  Country normalization: {time.time() - t_sub:.1f}s")

    # Keep required normalized columns
    keep_cols = ["entity_id", "name_norm", "name_clean_legal", "address_norm", "country_norm"]
    df = df[keep_cols]

    # Save to parquet
    t_sub = time.time()
    df.to_parquet(target_parquet, index=False, engine="pyarrow", compression="snappy")
    elapsed = time.time() - t0
    print(f"Saved {n_rows:,} rows to {target_parquet.name} ({elapsed:.1f}s total)")

    del df
    gc.collect()
    return target_parquet


def main():
    start_all = time.time()
    print("=" * 75)
    print("STEP 1: TEST DATA NORMALIZATION")
    print("=" * 75)
    print(f"Test Input Dir : {TEST_DIR}")
    print(f"Normalized Out : {OUTPUT_DIR}")

    normalize_source_file("test_source1.tsv", "s1_test_normalized.parquet", "Source 1")
    normalize_source_file("test_source2.tsv", "s2_test_normalized.parquet", "Source 2")
    normalize_source_file("test_source3.tsv", "s3_test_normalized.parquet", "Source 3")

    total_min = (time.time() - start_all) / 60
    print(f"\n{'=' * 75}")
    print(f"STEP 1 COMPLETE in {total_min:.2f} mins")
    print("=" * 75)


if __name__ == "__main__":
    main()
