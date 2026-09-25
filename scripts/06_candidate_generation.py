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

# Reconfigure stdout for UTF-8 line buffering on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Determine repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.preprocessing.normalization import normalize_series_fast

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
    print("Loading raw datasets...")
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

    print("Normalizing Source 1...")
    s1["name_norm"], s1["name_clean_legal"] = normalize_series_fast(s1["business_name"], is_name=True)
    s1["address_norm"], _ = normalize_series_fast(s1["business_address"], is_name=False)

    print("Normalizing Source 2...")
    s2["name_norm"], s2["name_clean_legal"] = normalize_series_fast(s2["business_name"], is_name=True)
    s2["address_norm"], _ = normalize_series_fast(s2["business_address"], is_name=False)

    print("Normalizing Source 3...")
    s3["name_norm"], s3["name_clean_legal"] = normalize_series_fast(s3["business_name"], is_name=True)
    s3["address_norm"], _ = normalize_series_fast(s3["business_address"], is_name=False)

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


# ============================================================
# Blocking Pass Algorithms
# ============================================================

def block_exact_field(s1_df: pd.DataFrame, other_df: pd.DataFrame, field: str, block_name: str) -> List[Tuple[str, str, str]]:
    """
    Exact string equality blocking pass on specified column field.
    """
    t0 = time.time()
    # Group other records by field
    other_valid = other_df[other_df[field] != ""]
    other_index = other_valid.groupby(field)["entity_id"].apply(list).to_dict()

    pairs = []
    s1_valid = s1_df[s1_df[field] != ""]
    for s1_id, val in zip(s1_valid["entity_id"], s1_valid[field]):
        if val in other_index:
            for ot_id in other_index[val]:
                pairs.append((s1_id, ot_id, block_name))

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(pairs):,} candidate pairs ({dt:.1f}s)")
    return pairs


def block_rare_tokens(s1_df: pd.DataFrame, other_df: pd.DataFrame, max_df: int = 500, max_cand_per_s1: int = 50) -> List[Tuple[str, str, str]]:
    """
    Informative Token Blocking:
    Indexes rare/informative tokens in `name_norm` below document frequency threshold `max_df`.
    """
    t0 = time.time()
    block_name = "rare_token"

    # Count token frequencies in other dataset
    token_counts = Counter()
    other_names = other_df[other_df["name_norm"] != ""]
    for name in other_names["name_norm"]:
        for token in set(name.split()):
            token_counts[token] += 1

    # Filter rare informative tokens (DF <= max_df and length >= 3)
    valid_tokens = {t for t, count in token_counts.items() if 1 <= count <= max_df and len(t) >= 3}

    # Build inverted index for other
    inverted_index = defaultdict(list)
    for ot_id, name in zip(other_names["entity_id"], other_names["name_norm"]):
        for token in set(name.split()):
            if token in valid_tokens:
                inverted_index[token].append(ot_id)

    pairs = []
    s1_names = s1_df[s1_df["name_norm"] != ""]
    for s1_id, name in zip(s1_names["entity_id"], s1_names["name_norm"]):
        cand_set = set()
        for token in set(name.split()):
            if token in inverted_index:
                cand_set.update(inverted_index[token])
                if len(cand_set) >= max_cand_per_s1:
                    break
        for ot_id in list(cand_set)[:max_cand_per_s1]:
            pairs.append((s1_id, ot_id, block_name))

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(pairs):,} candidate pairs (DF <= {max_df}, {dt:.1f}s)")
    return pairs


def block_address_tokens(s1_df: pd.DataFrame, other_df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """
    Address Component Blocking:
    Extracts house/building numbers, postal codes, and distinctive address tokens.
    """
    t0 = time.time()
    block_name = "address_component"

    def extract_address_keys(addr: str) -> List[str]:
        if not addr:
            return []
        tokens = addr.split()
        # Find numeric / alphanumeric house numbers or zip codes (digits present)
        num_keys = [t for t in tokens if any(c.isdigit() for c in t) and len(t) >= 2]
        return num_keys

    # Count address key frequency in other dataset
    other_addrs = other_df[other_df["address_norm"] != ""]
    key_counts = Counter()
    for addr in other_addrs["address_norm"]:
        for k in set(extract_address_keys(addr)):
            key_counts[k] += 1

    # Keep informative address keys (DF <= 1000)
    valid_keys = {k for k, count in key_counts.items() if 1 <= count <= 1000}

    inverted_index = defaultdict(list)
    for ot_id, addr in zip(other_addrs["entity_id"], other_addrs["address_norm"]):
        for k in set(extract_address_keys(addr)):
            if k in valid_keys:
                inverted_index[k].append(ot_id)

    pairs = []
    s1_addrs = s1_df[s1_df["address_norm"] != ""]
    for s1_id, addr in zip(s1_addrs["entity_id"], s1_addrs["address_norm"]):
        cands = set()
        keys = extract_address_keys(addr)
        if len(keys) >= 2:
            # Require sharing at least 2 distinct address numeric/code keys for precision
            matched_counts = Counter()
            for k in set(keys):
                if k in inverted_index:
                    for ot_id in inverted_index[k]:
                        matched_counts[ot_id] += 1
            cands = {ot_id for ot_id, cnt in matched_counts.items() if cnt >= 2}
        elif len(keys) == 1:
            k = keys[0]
            if k in inverted_index and len(inverted_index[k]) <= 50:
                cands = set(inverted_index[k])

        for ot_id in list(cands)[:30]:
            pairs.append((s1_id, ot_id, block_name))

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(pairs):,} candidate pairs ({dt:.1f}s)")
    return pairs


def block_tfidf_char_ngram(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    min_sim: float = 0.70,
    top_k: int = 10,
    sample_limit: int = 200000,
) -> List[Tuple[str, str, str]]:
    """
    TF-IDF Character 3-gram Nearest Neighbor Retrieval:
    Captures character-level typos and spelling mutations.
    """
    t0 = time.time()
    block_name = "char_ngram"

    # Filter non-empty names
    s1_valid = s1_df[s1_df["name_clean_legal"] != ""].copy()
    other_valid = other_df[other_df["name_clean_legal"] != ""].copy()

    # Apply sampling for performance if dataset exceeds limit
    if len(s1_valid) > sample_limit:
        s1_sub = s1_valid.head(sample_limit)
    else:
        s1_sub = s1_valid

    if len(other_valid) > sample_limit:
        other_sub = other_valid.head(sample_limit)
    else:
        other_sub = other_valid

    # Fit TfidfVectorizer on char 3-grams
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2)
    corpus = pd.concat([s1_sub["name_clean_legal"], other_sub["name_clean_legal"]])
    vectorizer.fit(corpus)

    X_s1 = vectorizer.transform(s1_sub["name_clean_legal"])
    X_other = vectorizer.transform(other_sub["name_clean_legal"])

    # Compute dot product (Cosine Similarity) in sparse chunks
    pairs = []
    chunk_size = 10000
    s1_ids = s1_sub["entity_id"].values
    other_ids = other_sub["entity_id"].values

    for start_idx in range(0, X_s1.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, X_s1.shape[0])
        sim_matrix = X_s1[start_idx:end_idx].dot(X_other.T)

        for row_i in range(sim_matrix.shape[0]):
            row = sim_matrix[row_i]
            col_indices = row.indices
            data = row.data

            # Filter candidates >= min_sim
            valid_mask = data >= min_sim
            if not np.any(valid_mask):
                continue

            valid_cols = col_indices[valid_mask]
            valid_sims = data[valid_mask]

            # Top-K candidates
            if len(valid_cols) > top_k:
                top_indices = np.argpartition(valid_sims, -top_k)[-top_k:]
                valid_cols = valid_cols[top_indices]

            s1_id = s1_ids[start_idx + row_i]
            for col_i in valid_cols:
                pairs.append((s1_id, other_ids[col_i], block_name))

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(pairs):,} candidate pairs (sim >= {min_sim}, {dt:.1f}s)")
    return pairs


# ============================================================
# Union, Evaluation & Statistics
# ============================================================

def combine_blocks_and_evaluate(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    total_true_pairs: int,
    all_blocks: List[List[Tuple[str, str, str]]],
    source_name: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    print("\n" + "=" * 70)
    print(f"EVALUATING BLOCKING PASSES & UNION FOR {source_name}")
    print("=" * 70)

    # Evaluate individual block recall & count
    block_records = []
    block_pair_sets: Dict[str, Set[Tuple[str, str]]] = {}

    for block_list in all_blocks:
        if not block_list:
            continue
        bname = block_list[0][2]
        pair_set = {(r[0], r[1]) for r in block_list}
        block_pair_sets[bname] = pair_set

        # Calculate True Positives for this block
        tp = sum(1 for s1_id, ot_id in pair_set if s1_id in gt_dict and ot_id in gt_dict[s1_id])
        recall = tp / total_true_pairs if total_true_pairs > 0 else 0.0
        avg_cand = len(pair_set) / len(s1_df)

        print(f"  {bname:<25s} | Recall: {recall*100:6.2f}% ({tp:>9,} TP) | Candidates: {len(pair_set):>10,} (avg {avg_cand:.2f}/S1)")

    # Union all blocks into a candidate mapping with pass bitmasks
    candidate_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for block_list in all_blocks:
        for s1_id, ot_id, bname in block_list:
            candidate_map[(s1_id, ot_id)].append(bname)

    # Create merged DataFrame
    records = []
    for (s1_id, ot_id), bnames in candidate_map.items():
        records.append({
            "s1_entity_id": s1_id,
            "candidate_entity_id": ot_id,
            "blocking_passes": "|".join(sorted(set(bnames))),
        })

    cand_df = pd.DataFrame(records)
    total_candidates = len(cand_df)

    # Unique contribution (recovered ONLY by this block)
    print("\nBlock Unique Contributions (True Pairs recovered ONLY by this block):")
    print("-" * 65)
    for bname, pset in block_pair_sets.items():
        other_sets = [s for name, s in block_pair_sets.items() if name != bname]
        if other_sets:
            unique_to_b = pset - set().union(*other_sets)
        else:
            unique_to_b = pset
        tp_unique = sum(1 for s1_id, ot_id in unique_to_b if s1_id in gt_dict and ot_id in gt_dict[s1_id])
        print(f"  {bname:<25s} | Unique TP Recovered: {tp_unique:>7,}")

    # Overall Union Evaluation
    total_tp = sum(1 for (s1_id, ot_id) in candidate_map.keys() if s1_id in gt_dict and ot_id in gt_dict[s1_id])
    union_recall = total_tp / total_true_pairs if total_true_pairs > 0 else 0.0

    # Entity-level complete coverage
    retrieved_by_s1 = defaultdict(set)
    for s1_id, ot_id in candidate_map.keys():
        retrieved_by_s1[s1_id].add(ot_id)

    full_coverage = 0
    partial_coverage = 0
    zero_coverage = 0

    for s1_id, true_set in gt_dict.items():
        retrieved = retrieved_by_s1.get(s1_id, set())
        matched = true_set & retrieved
        if len(matched) == len(true_set):
            full_coverage += 1
        elif len(matched) > 0:
            partial_coverage += 1
        else:
            zero_coverage += 1

    total_gt_s1 = len(gt_dict)
    full_cov_pct = full_coverage / total_gt_s1 * 100
    zero_cov_pct = zero_coverage / total_gt_s1 * 100

    # Reduction ratio
    naive_search_space = float(len(s1_df)) * float(len(other_df))
    reduction_ratio = 1.0 - (total_candidates / naive_search_space)

    print("\n" + "-" * 65)
    print(f"UNION CANDIDATE RECALL: {union_recall*100:6.2f}% ({total_tp:,} / {total_true_pairs:,} True Positives)")
    print(f"Total Candidates Generated  : {total_candidates:,}")
    print(f"Mean Candidates per S1      : {total_candidates / len(s1_df):.2f}")
    print(f"Complete S1 Entity Recall   : {full_cov_pct:.2f}% ({full_coverage:,} S1 entities)")
    print(f"Zero True-Match Retrieval   : {zero_cov_pct:.2f}% ({zero_coverage:,} S1 entities)")
    print(f"Search Space Reduction Ratio: {reduction_ratio*100:.6f}%")
    print("-" * 65)

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
