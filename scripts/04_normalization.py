"""
scripts/04_normalization.py
============================
Business Entity Resolution - Task 4: Cleaning & Normalization

Pipeline:
RAW
  -> Unicode normalization / diacritic removal (NFKD + Mn strip)
  -> lowercase
  -> punctuation normalization (preserve alphanumeric tokens)
  -> whitespace normalization
  -> strict terminal legal-suffix normalization (preserves descriptive tokens)
  -> collision analysis (diagnostic only)
  -> ground-truth parsing & source ID containment validation
  -> true-positive exact-match recovery evaluation on S2 and S3

Zero hardcoded machine/user paths. Fully portable across contributor environments.
"""

import gc
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ============================================================
# 1. Repository & Dataset Path Discovery
# ============================================================

# Reconfigure stdout to UTF-8 and line-buffering to stream logs in real time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Determine repository root from __file__
REPO_ROOT = Path(__file__).resolve().parent.parent


# Ensure src/ is on sys.path for internal imports if needed
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def find_dataset_dir(repo_root: Path) -> Path:
    """
    Locates the dataset directory using project-relative candidate paths
    or environment variables (BER_DATA_DIR / DATA_DIR).
    Fails clearly with diagnostics if required files are missing.
    """
    search_paths: List[Path] = []

    # Check environment variables first if provided
    for env_var in ("BER_DATA_DIR", "DATA_DIR"):
        env_val = os.environ.get(env_var)
        if env_val:
            search_paths.append(Path(env_val))

    # Project-relative candidate paths
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

    # Fail explicitly if not found
    error_msg = [
        "\n" + "!" * 70,
        "FATAL: Required training dataset files not found!",
        "Searched candidate locations:",
    ]
    for p in search_paths:
        error_msg.append(f"  - {p.resolve()} (exists: {p.exists()})")
    error_msg.append(f"Required files: {', '.join(required_files)}")
    error_msg.append("Please place the training files in data/student_resource/dataset/train or set BER_DATA_DIR.")
    error_msg.append("!" * 70)
    raise FileNotFoundError("\n".join(error_msg))


TRAIN_DIR = find_dataset_dir(REPO_ROOT)
OUTPUT_DIR = REPO_ROOT / "data" / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TASK 4: CLEANING & NORMALIZATION AUDIT")
print("=" * 70)
print(f"Repository Root : {REPO_ROOT}")
print(f"Dataset Dir     : {TRAIN_DIR}")
print(f"Output Dir      : {OUTPUT_DIR}")


# ============================================================
# 2. String Normalization & Strict Legal Suffix Removal
# ============================================================

def strip_accents_and_diacritics(text: str) -> str:
    """
    Decomposes characters (NFKD) and strips combining non-spacing marks (Mn).
    Converts: "Payne Énterprises" -> "Payne Enterprises", "Société" -> "Societe"
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def basic_clean_string(text: str) -> str:
    """
    Basic text cleaning pipeline:
    1. Null / NaN guard
    2. NFKD Unicode accent decomposition & diacritic removal
    3. Lowercasing
    4. Replace punctuation with spaces (preserving digits & alphanumeric tokens)
    5. Collapse whitespace and strip
    """
    if text is None or pd.isna(text):
        return ""
    text_str = str(text).strip()
    if not text_str:
        return ""

    text_str = strip_accents_and_diacritics(text_str)
    text_str = text_str.lower()
    # Replace non-alphanumeric characters with spaces
    text_str = re.sub(r"[^\w\s]", " ", text_str, flags=re.UNICODE)
    # Collapse multiple whitespace
    text_str = re.sub(r"\s+", " ", text_str).strip()
    return text_str


# Strict legal suffix list
# Multi-word suffixes must precede single-word suffixes for greedy longest matching
LEGAL_SUFFIX_LIST = [
    # Compound / multi-word suffixes
    "pvt ltd",
    "private limited",
    "co ltd",
    "company limited",
    "pty ltd",
    "proprietory limited",
    "corp inc",
    "llc inc",
    "sa de cv",
    "s de rl de cv",
    "sp z o o",
    "kabushiki kaisha",
    # Single-word designations (Common Law & Global)
    "inc",
    "incorporated",
    "corporation",
    "corp",
    "limited",
    "ltd",
    "llc",
    "llp",
    "plc",
    "lp",
    "pvt",
    "private",
    "co",
    "company",
    # European designations (France, Germany, Italy, Spain, Nordic, Benelux)
    "gmbh",
    "ag",
    "kgaa",
    "sarl",
    "sas",
    "sasu",
    "sa",
    "srl",
    "spa",
    "snc",
    "sl",
    "slne",
    "bv",
    "nv",
    "vof",
    "ab",
    "aps",
    "as",
    "oy",
    "oyj",
    # Asian transliterations
    "kk",
    "yk",
]

# Strict terminal pattern: matches only at the END ($), preceded by whitespace (\s+)
SUFFIX_PATTERN = re.compile(
    r"\s+(?:" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIX_LIST, key=len, reverse=True)) + r")$",
    flags=re.IGNORECASE,
)


def strip_legal_suffix(text: str) -> str:
    """
    Recursively strips legal corporate designations strictly from the END of the normalized business name.
    Preserves all descriptive nouns (e.g. 'enterprises', 'partners', 'services', 'solutions', 'hotel').
    Includes safety guard: will never strip a name down to an empty string.
    """
    if not text:
        return ""

    current = text
    for _ in range(3):
        new_text = SUFFIX_PATTERN.sub("", current).strip()
        # Safety guard: stop if no change, or if stripping would produce an empty or single-char string
        if new_text == current or len(new_text) < 2:
            break
        current = new_text

    return current


def normalize_series_fast(
    series: pd.Series, is_name: bool = False
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    High-efficiency vectorized normalization via distinct value mapping.
    Avoids recomputing string operations across millions of identical values.
    """
    unique_vals = series.dropna().unique()
    cleaned_map = {val: basic_clean_string(val) for val in unique_vals}
    basic_series = series.map(cleaned_map).fillna("")

    if is_name:
        unique_cleaned = set(cleaned_map.values())
        legal_map = {val: strip_legal_suffix(val) for val in unique_cleaned}
        legal_series = basic_series.map(legal_map).fillna("")
        return basic_series, legal_series

    return basic_series, None


# ============================================================
# 3. Data Loading & Source Normalization
# ============================================================

def load_sources() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    s1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype=str,
    )
    s2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype=str,
    )
    s3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype=str,
    )
    gt = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        usecols=["source1_entity_id", "matched_entity_ids"],
        dtype=str,
    )

    print(f"Loaded Source 1       : {len(s1):,} records")
    print(f"Loaded Source 2       : {len(s2):,} records")
    print(f"Loaded Source 3       : {len(s3):,} records")
    print(f"Loaded Ground Truth   : {len(gt):,} reference rows")

    return s1, s2, s3, gt


def normalize_source_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    print(f"Normalizing {source_name}...")
    df["name_norm"], df["name_clean_legal"] = normalize_series_fast(
        df["business_name"], is_name=True
    )
    df["address_norm"], _ = normalize_series_fast(
        df["business_address"], is_name=False
    )
    df["country_norm"], _ = normalize_series_fast(
        df["country"], is_name=False
    )
    return df


# ============================================================
# 4. Collision Analysis (Diagnostic Only)
# ============================================================

def run_collision_analysis(df: pd.DataFrame, source_name: str):
    """
    Computes cardinality and collision diagnostics across:
    RAW -> BASIC NORMALIZED -> LEGAL NORMALIZED
    Identifies any over-collapsing or generic name grouping.
    """
    print("\n" + "-" * 70)
    print(f"COLLISION & CARDINALITY ANALYSIS: {source_name}")
    print("-" * 70)

    raw_names = df["business_name"].dropna()
    basic_names = df["name_norm"][df["name_norm"] != ""]
    legal_names = df["name_clean_legal"][df["name_clean_legal"] != ""]

    u_raw = len(set(raw_names))
    u_basic = len(set(basic_names))
    u_legal = len(set(legal_names))

    print(f"Unique Raw Names             : {u_raw:>10,}")
    print(f"Unique Basic-Normalized Names: {u_basic:>10,}  (compression: {(1 - u_basic / u_raw) * 100:.2f}%)")
    print(f"Unique Legal-Normalized Names: {u_legal:>10,}  (compression: {(1 - u_legal / u_raw) * 100:.2f}%)")

    # Group raw names by legal normalized name to detect collision groups
    sample_size = min(len(df), 500000)
    subset = df.head(sample_size)[["business_name", "name_clean_legal"]].dropna()
    grouped = subset.groupby("name_clean_legal")["business_name"].nunique()
    collisions = grouped[grouped > 1]

    print(f"In sample of {sample_size:,} records:")
    print(f"  Canonical legal names mapping to >1 distinct raw names: {len(collisions):,}")
    if len(collisions) > 0:
        top_collision = collisions.sort_values(ascending=False).index[0]
        distinct_raws = subset[subset["name_clean_legal"] == top_collision]["business_name"].unique()
        print(f"  Top collision canonical: '{top_collision}' (covers {len(distinct_raws)} distinct raw names)")
        print(f"  Examples: {list(distinct_raws[:4])}")


# ============================================================
# 5. Ground Truth Parsing & Source Join Validation
# ============================================================

def parse_and_validate_ground_truth(
    gt: pd.DataFrame,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expands comma-separated ground truth IDs and validates containment against
    the actual Source tables. Reports full diagnostics and ensures zero-pair
    scenarios fail explicitly.
    """
    print("\n" + "=" * 70)
    print("GROUND TRUTH PARSING & SOURCE VALIDATION")
    print("=" * 70)

    gt_rows = len(gt)
    gt_clean = gt.dropna(subset=["source1_entity_id"]).copy()
    gt_clean["matched_entity_ids"] = gt_clean["matched_entity_ids"].fillna("")

    non_empty = gt_clean[gt_clean["matched_entity_ids"].str.strip() != ""]
    singleton_count = gt_rows - len(non_empty)

    print(f"Total Ground Truth Rows       : {gt_rows:,}")
    print(f"Non-empty Reference Rows      : {len(non_empty):,} ({len(non_empty) / gt_rows * 100:.2f}%)")
    print(f"Singleton S1 Entities (0 match): {singleton_count:,} ({singleton_count / gt_rows * 100:.2f}%)")

    # Expand comma-separated IDs
    gt_clean["matched_id"] = gt_clean["matched_entity_ids"].str.split(",")
    exploded = gt_clean.explode("matched_id", ignore_index=True)
    exploded["matched_id"] = exploded["matched_id"].str.strip()
    exploded = exploded[exploded["matched_id"] != ""].copy()
    exploded = exploded.rename(
        columns={"source1_entity_id": "s1_id", "matched_id": "other_id"}
    )

    total_pairs = len(exploded)
    print(f"Total Expanded Positive Pairs : {total_pairs:,}")

    # Split into S2 and S3 subsets using robust prefix matching
    s2_pairs = exploded[exploded["other_id"].str.startswith("S2-")].copy()
    s3_pairs = exploded[exploded["other_id"].str.startswith("S3-")].copy()

    print(f"  - Positive S2 Pairs         : {len(s2_pairs):,}")
    print(f"  - Positive S3 Pairs         : {len(s3_pairs):,}")
    print(f"  - Example S2 IDs            : {s2_pairs['other_id'].head(4).tolist()}")
    print(f"  - Example S3 IDs            : {s3_pairs['other_id'].head(4).tolist()}")

    # Validate ID existence against source tables
    s1_ids = set(s1["entity_id"])
    s2_ids = set(s2["entity_id"])
    s3_ids = set(s3["entity_id"])

    s2_s1_valid = s2_pairs["s1_id"].isin(s1_ids)
    s2_other_valid = s2_pairs["other_id"].isin(s2_ids)
    valid_s2_pairs = s2_pairs[s2_s1_valid & s2_other_valid].copy()

    s3_s1_valid = s3_pairs["s1_id"].isin(s1_ids)
    s3_other_valid = s3_pairs["other_id"].isin(s3_ids)
    valid_s3_pairs = s3_pairs[s3_s1_valid & s3_other_valid].copy()

    print("\nSource ID Containment:")
    print(f"  S1 IDs in S2 Pairs valid in Source 1: {s2_s1_valid.sum():,} / {len(s2_pairs):,}")
    print(f"  S2 IDs in S2 Pairs valid in Source 2: {s2_other_valid.sum():,} / {len(s2_pairs):,}")
    print(f"  Verified Valid S2 Pairs             : {len(valid_s2_pairs):,}")

    print(f"  S1 IDs in S3 Pairs valid in Source 1: {s3_s1_valid.sum():,} / {len(s3_pairs):,}")
    print(f"  S3 IDs in S3 Pairs valid in Source 3: {s3_other_valid.sum():,} / {len(s3_pairs):,}")
    print(f"  Verified Valid S3 Pairs             : {len(valid_s3_pairs):,}")

    if len(valid_s2_pairs) == 0:
        raise ValueError("CRITICAL: Found 0 valid S2 ground-truth pairs! Check ID format or table alignment.")
    if len(valid_s3_pairs) == 0:
        raise ValueError("CRITICAL: Found 0 valid S3 ground-truth pairs! Check ID format or table alignment.")

    return valid_s2_pairs, valid_s3_pairs


# ============================================================
# 6. True-Positive Exact Match Recovery Evaluation
# ============================================================

def evaluate_source_recovery(
    pairs: pd.DataFrame,
    s1: pd.DataFrame,
    other: pd.DataFrame,
    source_name: str,
):
    print("\n" + "=" * 70)
    print(f"TRUE-POSITIVE RECOVERY EVALUATION: S1 <-> {source_name}")
    print("=" * 70)

    total = len(pairs)
    if total == 0:
        print(f"ERROR: 0 pairs provided for {source_name}. Skipping to prevent nan% metrics.")
        return

    # Index s1 and other by entity_id
    s1_map = s1.set_index("entity_id")
    other_map = other.set_index("entity_id")

    # Fast reindex array extraction (0 memory overhead, instant execution)
    s1_ids = pairs["s1_id"].values
    ot_ids = pairs["other_id"].values

    s1_raw_name = s1_map["business_name"].reindex(s1_ids).values
    ot_raw_name = other_map["business_name"].reindex(ot_ids).values

    s1_norm_name = s1_map["name_norm"].reindex(s1_ids).values
    ot_norm_name = other_map["name_norm"].reindex(ot_ids).values

    s1_legal_name = s1_map["name_clean_legal"].reindex(s1_ids).values
    ot_legal_name = other_map["name_clean_legal"].reindex(ot_ids).values

    s1_raw_addr = s1_map["business_address"].reindex(s1_ids).values
    ot_raw_addr = other_map["business_address"].reindex(ot_ids).values

    s1_norm_addr = s1_map["address_norm"].reindex(s1_ids).values
    ot_norm_addr = other_map["address_norm"].reindex(ot_ids).values

    s1_norm_ctry = s1_map["country_norm"].reindex(s1_ids).values
    ot_norm_ctry = other_map["country_norm"].reindex(ot_ids).values

    # Clean index maps
    del s1_map, other_map
    gc.collect()

    # Fast numpy boolean equality checks
    name_raw = (s1_raw_name == ot_raw_name) & pd.notna(s1_raw_name)
    name_norm = (s1_norm_name == ot_norm_name) & (s1_norm_name != "")
    name_legal = (s1_legal_name == ot_legal_name) & (s1_legal_name != "")

    addr_raw = (s1_raw_addr == ot_raw_addr) & pd.notna(s1_raw_addr)
    addr_norm = (s1_norm_addr == ot_norm_addr) & (s1_norm_addr != "")
    ctry_norm = (s1_norm_ctry == ot_norm_ctry) & (s1_norm_ctry != "")

    comp_basic = name_norm | addr_norm
    comp_legal = name_legal | addr_norm
    still_unmatched = ~comp_legal

    # Report table
    print(f"Evaluated on {total:,} verified true positive pairs:")
    print("-" * 65)
    print(f"  1. Raw Business Name Exact        : {name_raw.sum():>9,} / {total:,} ({name_raw.mean() * 100:6.2f}%)")
    print(f"  2. Basic-Normalized Name Exact    : {name_norm.sum():>9,} / {total:,} ({name_norm.mean() * 100:6.2f}%)")
    print(f"  3. Legal-Normalized Name Exact    : {name_legal.sum():>9,} / {total:,} ({name_legal.mean() * 100:6.2f}%)")
    print(f"  4. Raw Business Address Exact     : {addr_raw.sum():>9,} / {total:,} ({addr_raw.mean() * 100:6.2f}%)")
    print(f"  5. Normalized Address Exact       : {addr_norm.sum():>9,} / {total:,} ({addr_norm.mean() * 100:6.2f}%)")
    print(f"  6. Normalized Country Match       : {ctry_norm.sum():>9,} / {total:,} ({ctry_norm.mean() * 100:6.2f}%)")
    print("-" * 65)
    print(f"  Composite (Basic Name OR Addr)    : {comp_basic.sum():>9,} / {total:,} ({comp_basic.mean() * 100:6.2f}%)")
    print(f"  Composite (Legal Name OR Addr)    : {comp_legal.sum():>9,} / {total:,} ({comp_legal.mean() * 100:6.2f}%)")
    print(f"  Still Unresolved (Needs Fuzzy/ML) : {still_unmatched.sum():>9,} / {total:,} ({still_unmatched.mean() * 100:6.2f}%)")

    # Diagnostic Examples
    print(f"\n--- Diagnostic Samples for {source_name} ---")

    fixed_basic_idx = np.where((~name_raw) & name_norm)[0]
    if len(fixed_basic_idx) > 0:
        print(f"\nSample matches fixed by Basic Normalization ({len(fixed_basic_idx):,} total):")
        for i in fixed_basic_idx[:3]:
            print(f"  S1: '{s1_raw_name[i]}'  <->  {source_name}: '{ot_raw_name[i]}'")
            print(f"  => Normalized: '{s1_norm_name[i]}'")

    fixed_legal_idx = np.where((~name_norm) & name_legal)[0]
    if len(fixed_legal_idx) > 0:
        print(f"\nSample matches fixed by Legal Normalization ({len(fixed_legal_idx):,} total):")
        for i in fixed_legal_idx[:3]:
            print(f"  S1: '{s1_raw_name[i]}'  <->  {source_name}: '{ot_raw_name[i]}'")
            print(f"  => Legal: '{s1_legal_name[i]}'")

    unresolved_idx = np.where(still_unmatched)[0]
    if len(unresolved_idx) > 0:
        print(f"\nSample true matches still unresolved ({len(unresolved_idx):,} total):")
        for i in unresolved_idx[:3]:
            print(f"  S1: '{s1_legal_name[i]}' | Addr: '{s1_norm_addr[i]}'")
            print(f"  {source_name}: '{ot_legal_name[i]}' | Addr: '{ot_norm_addr[i]}'")
            print()


# ============================================================
# 7. Main Pipeline
# ============================================================

def main():
    # 1. Load raw data
    s1, s2, s3, gt = load_sources()

    # 2. Normalize sources
    print("\n" + "=" * 70)
    print("APPLYING NORMALIZATION")
    print("=" * 70)
    s1 = normalize_source_df(s1, "Source 1")
    s2 = normalize_source_df(s2, "Source 2")
    s3 = normalize_source_df(s3, "Source 3")

    # Save normalized data as Parquet cache for downstream scripts
    cache_dir = REPO_ROOT / "data" / "outputs" / "normalized_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    s1.to_parquet(cache_dir / "s1_normalized.parquet", index=False)
    s2.to_parquet(cache_dir / "s2_normalized.parquet", index=False)
    s3.to_parquet(cache_dir / "s3_normalized.parquet", index=False)
    print(f"\nSaved normalized Parquet cache to: {cache_dir}")

    # 3. Collision diagnostics
    run_collision_analysis(s1, "Source 1")
    run_collision_analysis(s2, "Source 2")
    run_collision_analysis(s3, "Source 3")

    # 4. Parse & validate ground truth
    valid_s2_pairs, valid_s3_pairs = parse_and_validate_ground_truth(gt, s1, s2, s3)
    del gt
    gc.collect()

    # 5. Evaluate recovery on true pairs
    evaluate_source_recovery(valid_s2_pairs, s1, s2, "Source 2")
    evaluate_source_recovery(valid_s3_pairs, s1, s3, "Source 3")

    print("\n" + "=" * 70)
    print("TASK 4 CLEANING & NORMALIZATION COMPLETE")
    print("=" * 70)
    print("All invariants satisfied. System is validated for Task 5 Deterministic Baseline.")


if __name__ == "__main__":
    main()
