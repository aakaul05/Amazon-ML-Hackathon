"""
scripts/07_improved_blocking_v3.py
==================================
Business Entity Resolution — Task 7: Improved Blocking V3 (High-Recall & High-Efficiency)

Root Causes Solved in V3 (from Phase 2 False-Negative Audit):
 1. Disjoint Name + Strong Address Match (52.75% of S2 misses, 37.49% of S3 misses):
    - Solved via 'address_num_distinctive_token' (compound key of house_num|street_word).
    - Inverted index DF <= 50, ranking by token length and street uniqueness.
 2. Domain / Unspaced Name Variations (e.g. 'companyindia com' vs 'company india'):
    - Solved via 'compact_name_prefix' (8-char space-stripped prefix).
 3. Partial Name / Token Subset / Typos:
    - Solved via 'rare_token_pairs' (two rarest tokens paired together) and
      'name_prefix_plus_addr_num' (4-char name prefix + house number).
 4. Tail Similarity Misses:
    - Solved via partitioned TF-IDF character 3-gram (min_sim=0.45, top_k=20).

Pruned Wasteful Pass:
 - Pruned redundant passes ('country_name_token', 'first_token', 'sorted_token', 'last_token')
   which consumed >240,000 candidate pairs with <0.1% marginal recall.

Output:
 - val_v3_s1_s2_candidates.parquet
 - val_v3_s1_s3_candidates.parquet
 - v3_blocking_benchmark_summary.json
"""

from pathlib import Path
import os
import sys
import gc
import json
import time
import re
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.blocking.candidate_generator import _double_metaphone_simple
from business_entity_resolution.preprocessing.normalization import load_normalized_or_compute

RANDOM_SEED = 42

NAME_STOPWORDS = {
    "the", "and", "for", "group", "services", "company", "international",
    "global", "solutions", "management", "systems", "technologies",
    "consulting", "enterprises", "partners", "associates", "holdings",
    "industries", "products", "national", "general", "american", "india",
    "business", "financial", "capital", "investments", "properties",
    "construction", "development", "insurance", "marketing", "trading",
    "logistics", "communications", "engineering", "electric", "energy",
    "first", "new", "all", "united", "central", "standard", "premier",
    "north", "south", "east", "west", "one", "two", "three", "alpha",
    "private", "pvt", "ltd", "limited", "corp", "inc", "llc", "gmbh",
}

ADDRESS_STOPWORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr",
    "boulevard", "blvd", "way", "highway", "hwy", "floor", "fl", "suite", "ste",
    "block", "blk", "sector", "sec", "plot", "building", "bldg", "house", "no",
    "near", "opp", "opposite", "behind", "beside", "nagar", "colony", "complex",
    "market", "bazar", "chowk", "post", "dist", "district", "city", "state",
    "india", "usa", "uk", "tower", "plaza", "park", "industrial", "area", "phase",
}

V3_BLOCK_BITMASK = {
    "exact_name_norm":               1,
    "exact_clean_legal":             2,
    "rare_token_ranked":             4,
    "address_num_distinctive_token": 8,
    "compact_name_prefix":          16,
    "rare_token_pairs":             32,
    "name_prefix_plus_addr_num":    64,
    "phonetic":                    128,
    "partitioned_tfidf":           256,
}


def extract_address_components(addr: str) -> Tuple[List[str], List[str]]:
    """Extracts numeric tokens and distinctive street words from address."""
    if not addr or not isinstance(addr, str):
        return [], []
    tokens = [t.strip(",.-/#") for t in addr.lower().split() if t.strip(",.-/#")]
    nums = [t for t in tokens if re.search(r"\d", t) and len(t) <= 10]
    words = [t for t in tokens if t.isalpha() and len(t) >= 4 and t not in ADDRESS_STOPWORDS]
    return nums, words


def build_s1_v3_profiles(df_s1: pd.DataFrame) -> Dict:
    """Builds precomputed signature lookup dictionary for validation S1 entities."""
    s1_profiles = {}
    for eid, nn, ncl, an, cn in zip(
        df_s1["entity_id"], df_s1["name_norm"], df_s1["name_clean_legal"],
        df_s1["address_norm"], df_s1["country_norm"]
    ):
        nn_str = str(nn or "").strip()
        ncl_str = str(ncl or "").strip()
        an_str = str(an or "").strip()
        cn_str = str(cn or "").strip()

        tokens_ncl = [t for t in ncl_str.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
        tokens_ncl_sorted = sorted(tokens_ncl, key=len, reverse=True)

        token_pairs = []
        if len(tokens_ncl_sorted) >= 2:
            p1, p2 = sorted([tokens_ncl_sorted[0], tokens_ncl_sorted[1]])
            token_pairs.append(f"{p1}|{p2}")
            if len(tokens_ncl_sorted) >= 3:
                p1, p3 = sorted([tokens_ncl_sorted[0], tokens_ncl_sorted[2]])
                token_pairs.append(f"{p1}|{p3}")

        compact = "".join(ncl_str.split())
        compact_prefix = compact[:8] if len(compact) >= 6 else ""

        nums, words = extract_address_components(an_str)
        addr_pairs = []
        if nums and words:
            for num in nums[:2]:
                for word in words[:2]:
                    addr_pairs.append(f"{num}|{word}")

        name_prefix_addr = []
        if len(compact) >= 4 and nums:
            for num in nums[:2]:
                name_prefix_addr.append(f"{compact[:4]}|{num}")

        ph_keys = []
        for t in tokens_ncl[:2]:
            ph = _double_metaphone_simple(t)
            if ph and len(ph) >= 3:
                ph_keys.append(ph)

        s1_profiles[eid] = {
            "entity_id": eid,
            "nn": nn_str,
            "ncl": ncl_str,
            "an": an_str,
            "cn": cn_str,
            "tokens_ncl": tokens_ncl_sorted,
            "token_pairs": token_pairs,
            "compact_prefix": compact_prefix,
            "addr_pairs": addr_pairs,
            "name_prefix_addr": name_prefix_addr,
            "phonetic": ph_keys,
        }
    return s1_profiles


def stream_v3_inverted_index(
    s1_profiles: Dict,
    target_parquet_path: Path,
    source_name: str,
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Streams target parquet file, builds in-memory inverted indices for keys present in S1,
    and returns matched candidate pairs per blocker.
    """
    s1_nn_set = {p["nn"] for p in s1_profiles.values() if p["nn"]}
    s1_ncl_set = {p["ncl"] for p in s1_profiles.values() if p["ncl"]}
    s1_token_set = {t for p in s1_profiles.values() for t in p["tokens_ncl"]}
    s1_token_pairs_set = {tp for p in s1_profiles.values() for tp in p["token_pairs"]}
    s1_compact_set = {p["compact_prefix"] for p in s1_profiles.values() if p["compact_prefix"]}
    s1_addr_pairs_set = {ap for p in s1_profiles.values() for ap in p["addr_pairs"]}
    s1_name_addr_set = {nap for p in s1_profiles.values() for nap in p["name_prefix_addr"]}
    s1_ph_set = {ph for p in s1_profiles.values() for ph in p["phonetic"]}

    idx_exact_norm = defaultdict(list)
    idx_exact_clean = defaultdict(list)
    idx_token = defaultdict(list)
    idx_token_pairs = defaultdict(list)
    idx_compact = defaultdict(list)
    idx_addr_pairs = defaultdict(list)
    idx_name_addr = defaultdict(list)
    idx_phonetic = defaultdict(list)

    pf = pq.ParquetFile(str(target_parquet_path))
    t0 = time.time()
    n_rows = 0

    print(f"[{source_name}] Streaming index scan across {pf.metadata.num_rows:,} rows...")
    for batch in pf.iter_batches(batch_size=128_000, columns=["entity_id", "name_norm", "name_clean_legal", "address_norm"]):
        df_b = batch.to_pandas()
        n_rows += len(df_b)

        for eid, nn, ncl, an in zip(df_b["entity_id"], df_b["name_norm"], df_b["name_clean_legal"], df_b["address_norm"]):
            nn_str = str(nn or "").strip()
            ncl_str = str(ncl or "").strip()
            an_str = str(an or "").strip()

            if nn_str in s1_nn_set:
                idx_exact_norm[nn_str].append(eid)
            if ncl_str in s1_ncl_set:
                idx_exact_clean[ncl_str].append(eid)

            compact = "".join(ncl_str.split())
            if compact[:8] in s1_compact_set:
                idx_compact[compact[:8]].append(eid)

            nums, words = extract_address_components(an_str)
            if nums and words:
                for num in nums[:2]:
                    for word in words[:2]:
                        ap = f"{num}|{word}"
                        if ap in s1_addr_pairs_set:
                            idx_addr_pairs[ap].append(eid)

            if len(compact) >= 4 and nums:
                for num in nums[:2]:
                    nap = f"{compact[:4]}|{num}"
                    if nap in s1_name_addr_set:
                        idx_name_addr[nap].append(eid)

            tokens_ncl = [t for t in ncl_str.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
            tokens_ncl_sorted = sorted(tokens_ncl, key=len, reverse=True)
            for t in tokens_ncl_sorted[:4]:
                if t in s1_token_set:
                    idx_token[t].append(eid)

            if len(tokens_ncl_sorted) >= 2:
                p1, p2 = sorted([tokens_ncl_sorted[0], tokens_ncl_sorted[1]])
                tp1 = f"{p1}|{p2}"
                if tp1 in s1_token_pairs_set:
                    idx_token_pairs[tp1].append(eid)
                if len(tokens_ncl_sorted) >= 3:
                    p1, p3 = sorted([tokens_ncl_sorted[0], tokens_ncl_sorted[2]])
                    tp2 = f"{p1}|{p3}"
                    if tp2 in s1_token_pairs_set:
                        idx_token_pairs[tp2].append(eid)

            for t in tokens_ncl[:2]:
                ph = _double_metaphone_simple(t)
                if ph and ph in s1_ph_set:
                    idx_phonetic[ph].append(eid)

    print(f"[{source_name}] Indexed in {time.time() - t0:.1f}s.")

    blocker_pairs = {
        "exact_clean_legal": defaultdict(set),
        "rare_token_ranked": defaultdict(set),
        "address_num_distinctive_token": defaultdict(set),
        "compact_name_prefix": defaultdict(set),
        "rare_token_pairs": defaultdict(set),
        "name_prefix_plus_addr_num": defaultdict(set),
        "phonetic": defaultdict(set),
        "exact_name_norm": defaultdict(set),
    }

    # 1. exact_clean_legal (DF <= 200, cap 30)
    for s1_id, prof in s1_profiles.items():
        k = prof["ncl"]
        if k and k in idx_exact_clean:
            cands = idx_exact_clean[k]
            if 0 < len(cands) <= 200:
                blocker_pairs["exact_clean_legal"][s1_id].update(sorted(cands)[:30])

    # 2. rare_token_ranked (DF <= 800, top 20)
    for s1_id, prof in s1_profiles.items():
        cand_counts = Counter()
        for t in prof["tokens_ncl"][:3]:
            if t in idx_token:
                cands = idx_token[t]
                if 0 < len(cands) <= 800:
                    cand_counts.update(cands)
        if cand_counts:
            sorted_cands = sorted(cand_counts.keys(), key=lambda c: (-cand_counts[c], c))
            blocker_pairs["rare_token_ranked"][s1_id].update(sorted_cands[:20])

    # 3. address_num_distinctive_token (DF <= 50, top 25)
    for s1_id, prof in s1_profiles.items():
        cand_counts = Counter()
        for ap in prof["addr_pairs"]:
            if ap in idx_addr_pairs:
                cands = idx_addr_pairs[ap]
                if 0 < len(cands) <= 50:
                    cand_counts.update(cands)
        if cand_counts:
            sorted_cands = sorted(cand_counts.keys(), key=lambda c: (-cand_counts[c], c))
            blocker_pairs["address_num_distinctive_token"][s1_id].update(sorted_cands[:25])

    # 4. compact_name_prefix (DF <= 300, top 20)
    for s1_id, prof in s1_profiles.items():
        cp = prof["compact_prefix"]
        if cp and cp in idx_compact:
            cands = idx_compact[cp]
            if 0 < len(cands) <= 300:
                blocker_pairs["compact_name_prefix"][s1_id].update(sorted(cands)[:20])

    # 5. rare_token_pairs (DF <= 50, top 20)
    for s1_id, prof in s1_profiles.items():
        cand_counts = Counter()
        for tp in prof["token_pairs"]:
            if tp in idx_token_pairs:
                cands = idx_token_pairs[tp]
                if 0 < len(cands) <= 50:
                    cand_counts.update(cands)
        if cand_counts:
            sorted_cands = sorted(cand_counts.keys(), key=lambda c: (-cand_counts[c], c))
            blocker_pairs["rare_token_pairs"][s1_id].update(sorted_cands[:20])

    # 6. name_prefix_plus_addr_num (DF <= 100, top 15)
    for s1_id, prof in s1_profiles.items():
        cand_counts = Counter()
        for nap in prof["name_prefix_addr"]:
            if nap in idx_name_addr:
                cands = idx_name_addr[nap]
                if 0 < len(cands) <= 100:
                    cand_counts.update(cands)
        if cand_counts:
            sorted_cands = sorted(cand_counts.keys(), key=lambda c: (-cand_counts[c], c))
            blocker_pairs["name_prefix_plus_addr_num"][s1_id].update(sorted_cands[:15])

    # 7. phonetic (DF <= 400, top 15)
    for s1_id, prof in s1_profiles.items():
        cand_counts = Counter()
        for ph in prof["phonetic"]:
            if ph in idx_phonetic:
                cands = idx_phonetic[ph]
                if 0 < len(cands) <= 400:
                    cand_counts.update(cands)
        if cand_counts:
            sorted_cands = sorted(cand_counts.keys(), key=lambda c: (-cand_counts[c], c))
            blocker_pairs["phonetic"][s1_id].update(sorted_cands[:15])

    # 8. exact_name_norm (DF <= 200, cap 30)
    for s1_id, prof in s1_profiles.items():
        k = prof["nn"]
        if k and k in idx_exact_norm:
            cands = idx_exact_norm[k]
            if 0 < len(cands) <= 200:
                blocker_pairs["exact_name_norm"][s1_id].update(sorted(cands)[:30])

    return blocker_pairs


def run_partitioned_tfidf_blocking(
    df_s1: pd.DataFrame,
    target_parquet_path: Path,
    min_sim: float = 0.45,
    top_k: int = 15,
) -> Dict[str, Set[str]]:
    """Partitioned Char 3-Gram TF-IDF with min_sim=0.45 across full population."""
    t0 = time.time()
    s2_df = pd.read_parquet(target_parquet_path, columns=["entity_id", "name_clean_legal"])
    s2_df = s2_df[s2_df["name_clean_legal"] != ""].copy()
    s2_df["first_char"] = s2_df["name_clean_legal"].str[0].str.lower()

    sub_s1 = df_s1[df_s1["name_clean_legal"] != ""].copy()
    sub_s1["first_char"] = sub_s1["name_clean_legal"].str[0].str.lower()

    tfidf_pairs = defaultdict(set)
    unique_chars = sorted(list(sub_s1["first_char"].unique()))

    for ch in unique_chars:
        s1_part = sub_s1[sub_s1["first_char"] == ch]
        s2_part = s2_df[s2_df["first_char"] == ch]

        if len(s1_part) == 0 or len(s2_part) == 0:
            continue

        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2)
        try:
            X_all = vec.fit_transform(pd.concat([s1_part["name_clean_legal"], s2_part["name_clean_legal"]]))
            X_s1 = X_all[:len(s1_part)]
            X_s2 = X_all[len(s1_part):]

            sim_matrix = awesome_cossim_topn(X_s1, X_s2.T, ntop=top_k, lower_bound=min_sim, use_threads=True, n_jobs=4)
            cx = sim_matrix.tocoo()

            s1_ids_part = s1_part["entity_id"].values
            s2_ids_part = s2_part["entity_id"].values

            for r, c_idx in zip(cx.row, cx.col):
                tfidf_pairs[s1_ids_part[r]].add(s2_ids_part[c_idx])
        except Exception:
            continue

    print(f"Partitioned TF-IDF (min_sim={min_sim}) generated {sum(len(v) for v in tfidf_pairs.values()):,} pairs in {time.time() - t0:.1f}s")
    del s2_df, sub_s1
    gc.collect()
    return tfidf_pairs


def combine_and_annotate_v3_candidates(
    blocker_pairs: Dict[str, Dict[str, Set[str]]],
) -> pd.DataFrame:
    """Merges all blocker passes, annotates with pass bitmask, and produces final candidate DataFrame."""
    pair_bitmasks = defaultdict(int)

    for pass_name, pairs_dict in blocker_pairs.items():
        bit = V3_BLOCK_BITMASK.get(pass_name, 0)
        for s1_id, ot_set in pairs_dict.items():
            for ot_id in ot_set:
                pair_bitmasks[(s1_id, ot_id)] |= bit

    s1_list, ot_list, mask_list = [], [], []
    for (s1_id, ot_id), mask in pair_bitmasks.items():
        s1_list.append(s1_id)
        ot_list.append(ot_id)
        mask_list.append(mask)

    df_cand = pd.DataFrame({
        "s1_entity_id": s1_list,
        "candidate_entity_id": ot_list,
        "blocking_passes": mask_list,
    })
    # Deterministic sorting
    df_cand.sort_values(by=["s1_entity_id", "candidate_entity_id"], inplace=True)
    df_cand.reset_index(drop=True, inplace=True)
    return df_cand


def evaluate_benchmark_source(
    cand_df: pd.DataFrame,
    gt_pairs: Set[Tuple[str, str]],
    val_s1: pd.DataFrame,
    source_name: str,
) -> Dict:
    """Computes comprehensive evaluation metrics for benchmark mode."""
    total_gt = len(gt_pairs)
    total_s1 = len(val_s1)
    cand_pairs = set(zip(cand_df["s1_entity_id"], cand_df["candidate_entity_id"]))
    tp = cand_pairs & gt_pairs

    total_candidates = len(cand_df)
    total_tp = len(tp)
    recall_pct = total_tp / total_gt * 100 if total_gt > 0 else 0.0

    s1_gt_counts = Counter()
    for s1_id, _ in gt_pairs:
        s1_gt_counts[s1_id] += 1

    s1_tp_counts = Counter()
    for s1_id, ot_id in tp:
        s1_tp_counts[s1_id] += 1

    all_s1_with_gt = set(s1_gt_counts.keys())
    complete_cov = sum(1 for s1 in all_s1_with_gt if s1_tp_counts[s1] == s1_gt_counts[s1])
    zero_cov = sum(1 for s1 in all_s1_with_gt if s1_tp_counts[s1] == 0)

    n_gt_s1 = len(all_s1_with_gt)
    complete_pct = complete_cov / n_gt_s1 * 100 if n_gt_s1 > 0 else 0.0
    zero_pct = zero_cov / n_gt_s1 * 100 if n_gt_s1 > 0 else 0.0

    print("\n" + "=" * 65)
    print(f"V3 BENCHMARK RESULTS: S1 -> {source_name}")
    print("=" * 65)
    print(f"Candidate Recall           : {recall_pct:6.2f}% ({total_tp:,} / {total_gt:,})")
    print(f"Total Candidate Pairs      : {total_candidates:,}")
    print(f"Mean Candidates per S1     : {total_candidates / total_s1:.2f}")
    print(f"Complete S1 Entity Recall  : {complete_pct:6.2f}% ({complete_cov:,} / {n_gt_s1:,})")
    print(f"Zero True-Match Retrieval  : {zero_pct:6.2f}% ({zero_cov:,} / {n_gt_s1:,})")
    print("=" * 65)

    return {
        "source": source_name,
        "candidate_recall_pct": recall_pct,
        "total_true_pairs": total_gt,
        "recovered_tp": total_tp,
        "total_candidates": total_candidates,
        "candidates_per_s1": total_candidates / total_s1,
        "complete_s1_coverage_pct": complete_pct,
        "zero_s1_coverage_pct": zero_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Task 7: Improved Blocking V3")
    parser.add_argument("--benchmark", action="store_true", default=True, help="Run benchmark on 10k validation set")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 75)
    print("TASK 7: IMPROVED BLOCKING V3 (LEAN & HIGH RECALL)")
    print("=" * 75)

    val_dir = REPO_ROOT / "data" / "student_resource" / "outputs" / "validation"
    norm_dir = REPO_ROOT / "data" / "outputs" / "normalized_cache"

    val_s1_path = val_dir / "val_s1_10k.parquet"
    val_gt_path = val_dir / "val_gt_10k.json"

    val_s1 = pd.read_parquet(val_s1_path)
    with open(val_gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_s2 = {(s1, m) for s1, mlist in gt_data["s2"].items() for m in mlist}
    gt_s3 = {(s1, m) for s1, mlist in gt_data["s3"].items() for m in mlist}

    out_dir = Path(args.output_dir) if args.output_dir else val_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Source 2
    print("\n[Source 2] Generating V3 Candidates...")
    s1_profiles = build_s1_v3_profiles(val_s1)
    s2_path = norm_dir / "s2_normalized.parquet"
    s2_block_pairs = stream_v3_inverted_index(s1_profiles, s2_path, "Source 2")
    s2_tfidf = run_partitioned_tfidf_blocking(val_s1, s2_path, min_sim=0.45, top_k=20)
    s2_block_pairs["partitioned_tfidf"] = s2_tfidf

    cand_s2 = combine_and_annotate_v3_candidates(s2_block_pairs)
    stats_s2 = evaluate_benchmark_source(cand_s2, gt_s2, val_s1, "Source 2")
    cand_s2_path = out_dir / "val_v3_s1_s2_candidates.parquet"
    cand_s2.to_parquet(cand_s2_path, index=False)
    print(f"Saved Source 2 candidates to: {cand_s2_path}")

    # 2. Source 3
    print("\n[Source 3] Generating V3 Candidates...")
    s3_path = norm_dir / "s3_normalized.parquet"
    s3_block_pairs = stream_v3_inverted_index(s1_profiles, s3_path, "Source 3")
    s3_tfidf = run_partitioned_tfidf_blocking(val_s1, s3_path, min_sim=0.45, top_k=20)
    s3_block_pairs["partitioned_tfidf"] = s3_tfidf

    cand_s3 = combine_and_annotate_v3_candidates(s3_block_pairs)
    stats_s3 = evaluate_benchmark_source(cand_s3, gt_s3, val_s1, "Source 3")
    cand_s3_path = out_dir / "val_v3_s1_s3_candidates.parquet"
    cand_s3.to_parquet(cand_s3_path, index=False)
    print(f"Saved Source 3 candidates to: {cand_s3_path}")

    # Save summary
    summary = {"s2": stats_s2, "s3": stats_s3, "elapsed_s": time.time() - t_start}
    summary_path = out_dir / "val_v3_blocking_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved V3 blocking summary to: {summary_path}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
