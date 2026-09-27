"""
scripts/test_pipeline_pilot.py
==============================
Deterministic 5,000-S1 Pilot for Test/Inference Pipeline.

Verifies the entire test inference pipeline on AWS EC2:
1. Test data loading (test_source1.tsv, test_source2.tsv, test_source3.tsv)
2. Normalization (US, India, France) using existing normalization.py
3. Task 7 10-Pass Blocking using exact successful EC2 configuration
4. Candidate union and blocking_passes bitmask encoding
5. 30 pairwise feature extraction using features.py (exact feature count and order)
6. Model loading (5 CatBoost fold models: catboost_matcher_fold1..5.cbm)
7. Ensemble probability computation (mean across 5 models)
8. Thresholding (>= 0.9800000190734863)
9. Candidate vs. Match subset relationship
10. Submission formatting (matching_results.tsv, candidate_pairs.tsv)
11. Official submission validation compatibility

Outputs saved to:
data/student_resource/outputs/test/pilot/
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Set up paths
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.preprocessing.normalization import (
    normalize_series_fast,
    basic_clean_string,
    strip_legal_suffix,
)
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
from business_entity_resolution.matching.features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    compute_features_batch,
)

# CatBoost classifier
try:
    # pyrefly: ignore [missing-import]
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# ============================================================
# PATH DISCOVERY & CONFIGURATION
# ============================================================

def find_test_dir() -> Path:
    """Locate the test dataset directory."""
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
        REPO_ROOT / "data" / "student_resource" / "dataset" / "test",
        Path("/home/ec2-user/Amazon-ML-Hackathon/dataset/student_resource/dataset/test"),
        Path.home() / "Amazon-ML-Hackathon" / "dataset" / "student_resource" / "dataset" / "test",
    ]

    for p in candidates:
        if p.is_dir() and (p / "test_source1.tsv").is_file():
            return p.resolve()

    raise FileNotFoundError(
        "Test dataset directory not found. Expected test_source1.tsv under dataset/student_resource/dataset/test."
    )


def find_models_dir() -> Path:
    """Locate the directory containing the 5 trained CatBoost models."""
    candidates = [
        REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models",
        Path("/home/ec2-user/Amazon-ML-Hackathon/data/student_resource/outputs/matching/models"),
        REPO_ROOT / "data" / "outputs" / "matching" / "models",
    ]
    for p in candidates:
        if p.is_dir() and (p / "catboost_matcher_fold1.cbm").is_file():
            return p.resolve()
    return REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models"


TEST_DIR = find_test_dir()
MODELS_DIR = find_models_dir()

OUTPUT_BASE = REPO_ROOT / "data" / "student_resource" / "outputs" / "test" / "pilot"
NORM_DIR = OUTPUT_BASE / "normalized"
BLOCK_DIR = OUTPUT_BASE / "blocking"
PRED_DIR = OUTPUT_BASE / "predictions"
SUBMISSION_DIR = OUTPUT_BASE / "submission"

for d in [NORM_DIR, BLOCK_DIR, PRED_DIR, SUBMISSION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FINAL_THRESHOLD = 0.9800000190734863
PILOT_S1_COUNT = 5000


# ============================================================
# HELPER: LOOKUP TABLE BUILDER
# ============================================================

def build_lookup_table(df_norm: pd.DataFrame) -> Dict[str, Tuple[str, str, str, str]]:
    """Builds fast tuple lookup: entity_id -> (name_norm, name_clean_legal, address_norm, country_norm)."""
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


# ============================================================
# HELPER: TEST CANDIDATE COMBINER (NO GROUND TRUTH NEEDED)
# ============================================================

def combine_test_blocks(block_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Unions candidates across blocking passes and encodes the bitmask
    into pipe-separated 'blocking_passes' string without requiring ground truth.
    Guarantees zero duplicate (s1_entity_id, candidate_entity_id) pairs.
    """
    valid_dfs = [df for df in block_dfs if df is not None and not df.empty]
    if not valid_dfs:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])

    # Ensure relaxed char ngram bitmask is supported
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
    cand_df = concat_df.groupby(
        ["s1_entity_id", "candidate_entity_id"], as_index=False
    )["mask"].sum()
    del concat_df
    gc.collect()

    cand_df["blocking_passes"] = cand_df["mask"].map(bitmask_to_str).fillna("")
    cand_df.drop(columns=["mask"], inplace=True)
    return cand_df


# ============================================================
# HELPER: 10 TASK-7 BLOCKING PASSES (SUCCESSFUL EC2 SETTINGS)
# ============================================================

def run_task7_blocks(s1_df: pd.DataFrame, other_df: pd.DataFrame, target_name: str) -> Tuple[List[pd.DataFrame], Dict[str, int]]:
    """
    Executes the 10 Task 7 blocking passes with the verified successful parameters:
      1. Exact name_norm
      2. Exact clean legal name
      3. Rare tokens: max_df=800, max_cand_per_s1=80
      4. Address component tokens
      5. TF-IDF char 3-gram: min_sim=0.70, top_k=10, sample_limit=500000
      6. TF-IDF char 3-gram relaxed: min_sim=0.45, top_k=20, sample_limit=500000
      7. Sorted tokens: max_key_df=500, max_cand_per_s1=50
      8. Name prefix (6 chars): max_key_df=200, max_cand_per_s1=30
      9. Country + Name token: max_key_df=500, max_cand_per_s1=60, min_token_len=4
      10. Phonetic (Double Metaphone): max_key_df=300, max_cand_per_s1=40
    """
    blocks = []
    pass_counts = {}

    def _execute(name: str, fn):
        t0 = time.time()
        res = fn()
        count = len(res) if res is not None else 0
        pass_counts[name] = count
        print(f"  [{name}] {count:,} pairs in {time.time() - t0:.1f}s")
        return res

    print(f"\n--- Running 10 Blocking Passes for S1 -> {target_name.upper()} ---")

    # Pass 1: Exact name_norm
    blocks.append(_execute("Pass 1: exact_name_norm", lambda: block_exact_field(s1_df, other_df, "name_norm", "exact_name_norm")))

    # Pass 2: Exact clean legal
    blocks.append(_execute("Pass 2: exact_clean_legal", lambda: block_exact_field(s1_df, other_df, "name_clean_legal", "exact_clean_legal")))

    # Pass 3: Rare tokens
    blocks.append(_execute("Pass 3: rare_token", lambda: block_rare_tokens(s1_df, other_df, max_df=800, max_cand_per_s1=80)))

    # Pass 4: Address tokens
    blocks.append(_execute("Pass 4: address_component", lambda: block_address_tokens(s1_df, other_df)))

    # Pass 5: TF-IDF char 3-gram (0.70)
    blocks.append(_execute("Pass 5: char_ngram_0.70", lambda: block_tfidf_char_ngram(s1_df, other_df, min_sim=0.70, top_k=10, sample_limit=500000)))

    # Pass 6: TF-IDF char 3-gram relaxed (0.45)
    blocks.append(_execute("Pass 6: char_ngram_0.45", lambda: block_tfidf_char_ngram(s1_df, other_df, min_sim=0.45, top_k=20, sample_limit=500000)))

    # Pass 7: Sorted-token
    blocks.append(_execute("Pass 7: sorted_token", lambda: block_sorted_tokens(s1_df, other_df, max_key_df=500, max_cand_per_s1=50)))

    # Pass 8: Name prefix (6 chars)
    blocks.append(_execute("Pass 8: name_prefix_6", lambda: block_name_prefix(s1_df, other_df, prefix_len=6, max_key_df=200, max_cand_per_s1=30)))

    # Pass 9: Country + Name compound
    blocks.append(_execute("Pass 9: country_name_token", lambda: block_country_name_token(s1_df, other_df, max_key_df=500, max_cand_per_s1=60, min_token_len=4)))

    # Pass 10: Phonetic
    blocks.append(_execute("Pass 10: phonetic", lambda: block_phonetic(s1_df, other_df, max_key_df=300, max_cand_per_s1=40)))

    return blocks, pass_counts


# ============================================================
# MAIN PILOT RUNNER
# ============================================================

def run_pilot():
    t_start = time.time()
    report = {}

    print("=" * 75)
    print("TEST PIPELINE PILOT (5,000 S1 ENTITIES)")
    print("=" * 75)
    print(f"Test Dataset Dir: {TEST_DIR}")
    print(f"Models Dir      : {MODELS_DIR}")
    print(f"Pilot Outputs   : {OUTPUT_BASE}")

    # ---------------------------------------------------------
    # Step 1: Load Test S1 Entities (Deterministic Sample)
    # ---------------------------------------------------------
    print("\n[Step 1] Loading deterministic 5,000 S1 test entities...")
    s1_cols = ["entity_id", "business_name", "business_address", "country"]
    s1_full = pd.read_csv(
        TEST_DIR / "test_source1.tsv",
        sep="\t",
        dtype=str,
        usecols=s1_cols,
        nrows=PILOT_S1_COUNT,
    )
    s1_full = s1_full.fillna("")
    print(f"Loaded {len(s1_full):,} S1 rows.")

    country_dist = s1_full["country"].value_counts().to_dict()
    print(f"S1 Country distribution: {country_dist}")
    report["s1_entities_processed"] = len(s1_full)
    report["s1_country_distribution"] = country_dist

    # ---------------------------------------------------------
    # Step 2: Normalization
    # ---------------------------------------------------------
    print("\n[Step 2] Normalizing S1 pilot sample...")
    s1_full["name_norm"], s1_full["name_clean_legal"] = normalize_series_fast(s1_full["business_name"], is_name=True)
    s1_full["address_norm"], _ = normalize_series_fast(s1_full["business_address"], is_name=False)
    s1_full["country_norm"], _ = normalize_series_fast(s1_full["country"], is_name=False)

    s1_pilot_parquet = NORM_DIR / "s1_pilot_normalized.parquet"
    s1_full.to_parquet(s1_pilot_parquet, index=False)
    print(f"Saved normalized S1 pilot to {s1_pilot_parquet.name}")

    # Build S1 lookup
    s1_lookup = build_lookup_table(s1_full)

    # Load and normalize a representative subset of S2 and S3 for blocking
    # (Reading first 100,000 rows of S2 and S3 for pilot blocking test)
    s2_pilot_raw = pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t", dtype=str, usecols=s1_cols, nrows=100000).fillna("")
    s2_pilot_raw["name_norm"], s2_pilot_raw["name_clean_legal"] = normalize_series_fast(s2_pilot_raw["business_name"], is_name=True)
    s2_pilot_raw["address_norm"], _ = normalize_series_fast(s2_pilot_raw["business_address"], is_name=False)
    s2_pilot_raw["country_norm"], _ = normalize_series_fast(s2_pilot_raw["country"], is_name=False)
    s2_lookup = build_lookup_table(s2_pilot_raw)
    report["s2_entities_considered"] = len(s2_pilot_raw)

    s3_pilot_raw = pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t", dtype=str, usecols=s1_cols, nrows=100000).fillna("")
    s3_pilot_raw["name_norm"], s3_pilot_raw["name_clean_legal"] = normalize_series_fast(s3_pilot_raw["business_name"], is_name=True)
    s3_pilot_raw["address_norm"], _ = normalize_series_fast(s3_pilot_raw["business_address"], is_name=False)
    s3_pilot_raw["country_norm"], _ = normalize_series_fast(s3_pilot_raw["country"], is_name=False)
    s3_lookup = build_lookup_table(s3_pilot_raw)
    report["s3_entities_considered"] = len(s3_pilot_raw)

    # ---------------------------------------------------------
    # Step 3: 10-Pass Blocking & Candidate Union
    # ---------------------------------------------------------
    print("\n[Step 3] Running 10-pass blocking for S1 -> S2...")
    s2_blocks, s2_pass_counts = run_task7_blocks(s1_full, s2_pilot_raw, "s2")
    s2_cands = combine_test_blocks(s2_blocks)
    s2_cands_path = BLOCK_DIR / "test_s1_s2_candidates_pilot.parquet"
    s2_cands.to_parquet(s2_cands_path, index=False)
    print(f"Total unique S1->S2 candidate pairs: {len(s2_cands):,}")
    report["s1_s2_candidate_count"] = len(s2_cands)
    report["s1_s2_pass_counts"] = s2_pass_counts

    print("\nRunning 10-pass blocking for S1 -> S3...")
    s3_blocks, s3_pass_counts = run_task7_blocks(s1_full, s3_pilot_raw, "s3")
    s3_cands = combine_test_blocks(s3_blocks)
    s3_cands_path = BLOCK_DIR / "test_s1_s3_candidates_pilot.parquet"
    s3_cands.to_parquet(s3_cands_path, index=False)
    print(f"Total unique S1->S3 candidate pairs: {len(s3_cands):,}")
    report["s1_s3_candidate_count"] = len(s3_cands)
    report["s1_s3_pass_counts"] = s3_pass_counts

    # Verify no duplicate candidate pairs
    s2_dups = len(s2_cands) - len(s2_cands[["s1_entity_id", "candidate_entity_id"]].drop_duplicates())
    s3_dups = len(s3_cands) - len(s3_cands[["s1_entity_id", "candidate_entity_id"]].drop_duplicates())
    report["duplicate_candidate_count"] = s2_dups + s3_dups
    print(f"Duplicate candidate check: S2={s2_dups}, S3={s3_dups} (must be 0)")

    # ---------------------------------------------------------
    # Step 4: 30-Feature Extraction
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Step 4: 30-Feature Extraction for S1->S2 and S1->S3
    # ---------------------------------------------------------
    print("\n[Step 4] Extracting 30 pairwise features for S1->S2 candidate pairs...")
    t_feat = time.time()
    n_pairs_s2 = len(s2_cands)

    s1_ids_s2 = s2_cands["s1_entity_id"].values
    cand_ids_s2 = s2_cands["candidate_entity_id"].values
    bp_s2 = s2_cands["blocking_passes"].values

    s1_n = [""] * n_pairs_s2
    s1_cl = [""] * n_pairs_s2
    s1_ad = [""] * n_pairs_s2
    s1_co = [""] * n_pairs_s2
    ot_n = [""] * n_pairs_s2
    ot_cl = [""] * n_pairs_s2
    ot_ad = [""] * n_pairs_s2
    ot_co = [""] * n_pairs_s2

    for i in range(n_pairs_s2):
        s1_d = s1_lookup.get(s1_ids_s2[i])
        if s1_d:
            s1_n[i], s1_cl[i], s1_ad[i], s1_co[i] = s1_d
        ot_d = s2_lookup.get(cand_ids_s2[i])
        if ot_d:
            ot_n[i], ot_cl[i], ot_ad[i], ot_co[i] = ot_d

    X_s2 = compute_features_batch(
        s1_names=s1_n, s1_clean=s1_cl, s1_addrs=s1_ad, s1_countries=s1_co,
        ot_names=ot_n, ot_clean=ot_cl, ot_addrs=ot_ad, ot_countries=ot_co,
        blocking_passes=bp_s2,
    )
    print(f"Extracted S1->S2 features matrix shape: {X_s2.shape} in {time.time() - t_feat:.2f}s")
    assert X_s2.shape[1] == NUM_FEATURES == 30, f"Expected 30 features, got {X_s2.shape[1]}"
    assert not np.isnan(X_s2).any(), "Found NaN in S2 feature matrix!"

    print("\nExtracting 30 pairwise features for S1->S3 candidate pairs...")
    t_feat3 = time.time()
    n_pairs_s3 = len(s3_cands)

    s1_ids_s3 = s3_cands["s1_entity_id"].values
    cand_ids_s3 = s3_cands["candidate_entity_id"].values
    bp_s3 = s3_cands["blocking_passes"].values

    s1_n3 = [""] * n_pairs_s3
    s1_cl3 = [""] * n_pairs_s3
    s1_ad3 = [""] * n_pairs_s3
    s1_co3 = [""] * n_pairs_s3
    ot_n3 = [""] * n_pairs_s3
    ot_cl3 = [""] * n_pairs_s3
    ot_ad3 = [""] * n_pairs_s3
    ot_co3 = [""] * n_pairs_s3

    for i in range(n_pairs_s3):
        s1_d = s1_lookup.get(s1_ids_s3[i])
        if s1_d:
            s1_n3[i], s1_cl3[i], s1_ad3[i], s1_co3[i] = s1_d
        ot_d = s3_lookup.get(cand_ids_s3[i])
        if ot_d:
            ot_n3[i], ot_cl3[i], ot_ad3[i], ot_co3[i] = ot_d

    X_s3 = compute_features_batch(
        s1_names=s1_n3, s1_clean=s1_cl3, s1_addrs=s1_ad3, s1_countries=s1_co3,
        ot_names=ot_n3, ot_clean=ot_cl3, ot_addrs=ot_ad3, ot_countries=ot_co3,
        blocking_passes=bp_s3,
    )
    print(f"Extracted S1->S3 features matrix shape: {X_s3.shape} in {time.time() - t_feat3:.2f}s")
    assert X_s3.shape[1] == NUM_FEATURES == 30, f"Expected 30 features, got {X_s3.shape[1]}"
    assert not np.isnan(X_s3).any(), "Found NaN in S3 feature matrix!"

    report["feature_count"] = 30
    report["exact_feature_names"] = FEATURE_NAMES

    # ---------------------------------------------------------
    # Step 5: Model Loading and 5-Model Ensemble Inference
    # ---------------------------------------------------------
    print("\n[Step 5] Checking and loading 5 CatBoost fold models...")
    models = []
    model_success = {}

    for fold in range(1, 6):
        m_file = MODELS_DIR / f"catboost_matcher_fold{fold}.cbm"
        if not m_file.is_file():
            print(f"  WARNING: Model file {m_file.name} not found in {MODELS_DIR}")
            model_success[f"model_{fold}_success"] = False
        else:
            if CATBOOST_AVAILABLE:
                m = CatBoostClassifier()
                m.load_model(str(m_file))
                models.append(m)
                model_success[f"model_{fold}_success"] = True
                print(f"  [Model {fold}/5] Loaded {m_file.name} successfully.")
            else:
                model_success[f"model_{fold}_success"] = "catboost_not_installed"

    report.update(model_success)

    # Score S1->S2 candidates
    if len(models) == 5:
        print("\nScoring S1->S2 candidates with 5-model ensemble...")
        fold_probs_s2 = [m.predict_proba(X_s2)[:, 1] for m in models]
        probs_s2 = np.mean(fold_probs_s2, axis=0)

        print("Scoring S1->S3 candidates with 5-model ensemble...")
        fold_probs_s3 = [m.predict_proba(X_s3)[:, 1] for m in models]
        probs_s3 = np.mean(fold_probs_s3, axis=0)

        all_probs = np.concatenate([probs_s2, probs_s3])
        report["probability_min"] = float(np.min(all_probs))
        report["probability_max"] = float(np.max(all_probs))
        report["probability_mean"] = float(np.mean(all_probs))

        report["number_above_threshold"] = int(np.sum(all_probs >= FINAL_THRESHOLD))
        report["number_below_threshold"] = int(np.sum(all_probs < FINAL_THRESHOLD))
    else:
        print(f"\nNote: {len(models)}/5 models loaded. (Full ensemble inference will run on EC2 where models exist).")
        probs_s2 = np.zeros(len(X_s2), dtype=np.float32)
        probs_s3 = np.zeros(len(X_s3), dtype=np.float32)
        report["probability_min"] = 0.0
        report["probability_max"] = 0.0
        report["probability_mean"] = 0.0
        report["number_above_threshold"] = 0
        report["number_below_threshold"] = len(probs_s2) + len(probs_s3)

    # Filter matches
    matched_s2_mask = probs_s2 >= FINAL_THRESHOLD
    matches_s2_df = pd.DataFrame({
        "s1_entity_id": s1_ids_s2[matched_s2_mask],
        "matched_entity_id": cand_ids_s2[matched_s2_mask],
        "probability": probs_s2[matched_s2_mask],
    })
    matches_s2_path = PRED_DIR / "test_s1_s2_matches_pilot.parquet"
    matches_s2_df.to_parquet(matches_s2_path, index=False)

    matched_s3_mask = probs_s3 >= FINAL_THRESHOLD
    matches_s3_df = pd.DataFrame({
        "s1_entity_id": s1_ids_s3[matched_s3_mask],
        "matched_entity_id": cand_ids_s3[matched_s3_mask],
        "probability": probs_s3[matched_s3_mask],
    })
    matches_s3_path = PRED_DIR / "test_s1_s3_matches_pilot.parquet"
    matches_s3_df.to_parquet(matches_s3_path, index=False)

    print(f"Saved {len(matches_s2_df):,} S2 matches and {len(matches_s3_df):,} S3 matches.")
    report["s2_matches"] = len(matches_s2_df)
    report["s3_matches"] = len(matches_s3_df)
    report["total_predicted_matches"] = len(matches_s2_df) + len(matches_s3_df)

    # ---------------------------------------------------------
    # Step 6: Create Pilot Submission TSVs
    # ---------------------------------------------------------
    print("\n[Step 6] Generating pilot submission files...")

    # Dictionary of S1 -> list of candidate IDs
    candidates_by_s1 = defaultdict(list)
    for s1, c in zip(s2_cands["s1_entity_id"], s2_cands["candidate_entity_id"]):
        candidates_by_s1[s1].append(c)
    for s1, c in zip(s3_cands["s1_entity_id"], s3_cands["candidate_entity_id"]):
        candidates_by_s1[s1].append(c)

    # Dictionary of S1 -> list of matched IDs
    matches_by_s1 = defaultdict(list)
    for s1, m in zip(matches_s2_df["s1_entity_id"], matches_s2_df["matched_entity_id"]):
        matches_by_s1[s1].append(m)
    for s1, m in zip(matches_s3_df["s1_entity_id"], matches_s3_df["matched_entity_id"]):
        matches_by_s1[s1].append(m)

    pilot_s1_list = list(s1_full["entity_id"])
    assert len(pilot_s1_list) == PILOT_S1_COUNT, f"Expected {PILOT_S1_COUNT} S1 entities"

    matching_rows = []
    candidate_rows = []
    multi_match_count = 0
    zero_match_count = 0

    for s1_id in pilot_s1_list:
        m_list = matches_by_s1.get(s1_id, [])
        c_list = candidates_by_s1.get(s1_id, [])

        if len(m_list) > 1:
            multi_match_count += 1
        elif len(m_list) == 0:
            zero_match_count += 1

        matching_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(m_list),
        })
        candidate_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(c_list),
        })

    report["number_multiple_matches"] = multi_match_count
    report["number_zero_matches"] = zero_match_count

    df_matching = pd.DataFrame(matching_rows)
    df_candidate = pd.DataFrame(candidate_rows)

    matching_tsv = SUBMISSION_DIR / "matching_results.tsv"
    candidate_tsv = SUBMISSION_DIR / "candidate_pairs.tsv"

    df_matching.to_csv(matching_tsv, sep="\t", index=False)
    df_candidate.to_csv(candidate_tsv, sep="\t", index=False)

    print(f"Generated {matching_tsv} ({len(df_matching):,} rows)")
    print(f"Generated {candidate_tsv} ({len(df_candidate):,} rows)")

    # ---------------------------------------------------------
    # Step 7: Integrity Checks and Validator Testing
    # ---------------------------------------------------------
    print("\n[Step 7] Running integrity verification...")
    # 1. Matching candidates are a strict subset of candidate candidates
    subset_violation = 0
    for s1_id in pilot_s1_list:
        m_set = set(matches_by_s1.get(s1_id, []))
        c_set = set(candidates_by_s1.get(s1_id, []))
        if not m_set.issubset(c_set):
            subset_violation += 1

    print(f"Subset check: matching <= candidates: {subset_violation == 0} (violations: {subset_violation})")
    assert subset_violation == 0, f"Found {subset_violation} entities where match not in candidates!"

    # 2. No NaN, exactly one row per S1
    assert len(df_matching) == PILOT_S1_COUNT
    assert len(df_candidate) == PILOT_S1_COUNT
    assert df_matching["source1_entity_id"].nunique() == PILOT_S1_COUNT

    # 3. France statistics
    france_s1_ids = set(s1_full[s1_full["country"].str.lower() == "france"]["entity_id"])
    france_matched = sum(1 for eid in france_s1_ids if len(matches_by_s1.get(eid, [])) > 0)
    report["france_s1_entities"] = len(france_s1_ids)
    report["france_s1_with_matches"] = france_matched

    # 4. Test official validator against mock pilot test directory
    mock_test_dir = OUTPUT_BASE / "mock_test_dir"
    mock_test_dir.mkdir(parents=True, exist_ok=True)
    s1_full[["entity_id", "business_name", "business_address", "country"]].to_csv(
        mock_test_dir / "test_source1.tsv", sep="\t", index=False
    )

    validator_script = REPO_ROOT / "dataset" / "student_resource" / "utils" / "validate_submission.py"
    if not validator_script.is_file():
        alt_val = REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
        if alt_val.is_file():
            validator_script = alt_val

    if validator_script.is_file():
        print(f"\nRunning official validator: {validator_script.name} on pilot output...")
        import subprocess
        cmd = [
            sys.executable,
            str(validator_script),
            "--matching", str(matching_tsv),
            "--candidate", str(candidate_tsv),
            "--test-dir", str(mock_test_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Validator Exit Code: {proc.returncode}")
        print(f"Validator STDOUT:\n{proc.stdout.strip()}")
        report["validator_exit_code"] = proc.returncode
        report["validator_status"] = "PASS" if proc.returncode == 0 else "FAIL"
    else:
        report["validator_status"] = "validator_script_not_found"

    # Sample rows for report
    sample_matched = df_matching[df_matching["matched_entity_ids"] != ""].head(5).to_dict(orient="records")
    sample_singletons = df_matching[df_matching["matched_entity_ids"] == ""].head(5)["source1_entity_id"].tolist()
    france_examples = df_matching[df_matching["source1_entity_id"].isin(france_s1_ids)].head(5).to_dict(orient="records")

    report["sample_matched_pairs"] = sample_matched
    report["sample_unmatched_s1_ids"] = sample_singletons
    report["france_examples"] = france_examples

    # Disk usage
    import shutil
    total, used, free = shutil.disk_usage(REPO_ROOT)
    report["disk_usage"] = {
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
    }

    report["runtime_seconds"] = round(time.time() - t_start, 2)

    # Save pilot report JSON
    report_file = OUTPUT_BASE / "pilot_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nPilot completed in {report['runtime_seconds']}s")
    print(f"Pilot report written to: {report_file}")
    print("=" * 75)
    return report


if __name__ == "__main__":
    run_pilot()
