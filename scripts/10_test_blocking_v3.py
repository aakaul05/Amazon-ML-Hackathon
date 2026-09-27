"""
scripts/10_test_blocking_v3.py
==============================
Step 2 of Test Pipeline: Test Candidate Generation using V3 Blocking (High-Recall & Lean)

Upgrades over old Test Blocking (V1):
 1. Address-Number + Distinctive-Token Blocking:
    - Captures pairs with distinct address numbers and street names even with completely disjoint names.
    - Recovered +30.5% marginal true matches in benchmark ablation.
 2. Space-Stripped Name Prefix (compact_name_prefix):
    - 8-char compact prefix resolves unspaced domain names (e.g. 'companyindia com' vs 'company india').
 3. Two Rare-Token Compound Pairs (rare_token_pairs):
    - Bypasses high-frequency single-token limits for multi-word corporate entities.
 4. Name-Prefix + Address-Number (name_prefix_plus_addr_num):
    - Recovers typo/abbreviated names sharing the exact building/house number.
 5. Partitioned TF-IDF (char 3-gram, min_sim=0.45):
    - Full-population coverage partitioned by first character (no 500k sampling truncation).
 6. Pruned Low-Yield Wasteful Pass:
    - Pruned country_name_token and first_token which caused candidate explosion without precision.

Outputs saved to:
  data/student_resource/outputs/test/blocking/
    - test_s1_s2_candidates.parquet
    - test_s1_s3_candidates.parquet
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

RANDOM_SEED = 42

NORM_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "test" / "normalized"
BLOCKING_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "test" / "blocking"
BLOCKING_DIR.mkdir(parents=True, exist_ok=True)

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
    """Builds precomputed signature lookup dictionary for test S1 entities."""
    s1_profiles = {}
    print(f"Building V3 lookup profiles for {len(df_s1):,} S1 entities...")
    t0 = time.time()
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
    print(f"Built profiles in {time.time() - t0:.1f}s")
    return s1_profiles


def stream_v3_inverted_index(
    s1_profiles: Dict,
    target_parquet_path: Path,
    source_name: str,
) -> Dict[str, Dict[str, Set[str]]]:
    """Streams target parquet file and builds matched candidate pairs per blocker."""
    print(f"[{source_name}] Building active key sets from S1...")
    t0 = time.time()
    s1_nn_set = {p["nn"] for p in s1_profiles.values() if p["nn"]}
    s1_ncl_set = {p["ncl"] for p in s1_profiles.values() if p["ncl"]}
    s1_token_set = {t for p in s1_profiles.values() for t in p["tokens_ncl"]}
    s1_token_pairs_set = {tp for p in s1_profiles.values() for tp in p["token_pairs"]}
    s1_compact_set = {p["compact_prefix"] for p in s1_profiles.values() if p["compact_prefix"]}
    s1_addr_pairs_set = {ap for p in s1_profiles.values() for ap in p["addr_pairs"]}
    s1_name_addr_set = {nap for p in s1_profiles.values() for nap in p["name_prefix_addr"]}
    s1_ph_set = {ph for p in s1_profiles.values() for ph in p["phonetic"]}
    print(f"[{source_name}] Active keys built in {time.time() - t0:.1f}s")

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


def combine_and_save_candidates(
    blocker_pairs: Dict[str, Dict[str, Set[str]]],
    out_parquet: Path,
    chunk_size: int = 1_000_000,
):
    """Merges all blocker passes, annotates bitmasks, and writes parquet in streaming chunks."""
    print(f"Combining and deduplicating candidate pairs...")
    t0 = time.time()
    pair_bitmasks = defaultdict(int)

    for pass_name, pairs_dict in blocker_pairs.items():
        bit = V3_BLOCK_BITMASK.get(pass_name, 0)
        for s1_id, ot_set in pairs_dict.items():
            for ot_id in ot_set:
                pair_bitmasks[(s1_id, ot_id)] |= bit

    total_pairs = len(pair_bitmasks)
    print(f"Total unique candidate pairs: {total_pairs:,} (deduplicated in {time.time() - t0:.1f}s)")

    s1_list, ot_list, mask_list = [], [], []
    for (s1_id, ot_id), mask in pair_bitmasks.items():
        s1_list.append(s1_id)
        ot_list.append(ot_id)
        mask_list.append(mask)

    del pair_bitmasks
    gc.collect()

    df_cand = pd.DataFrame({
        "s1_entity_id": s1_list,
        "candidate_entity_id": ot_list,
        "blocking_passes": mask_list,
    })
    del s1_list, ot_list, mask_list
    gc.collect()

    print("Sorting deterministically...")
    df_cand.sort_values(by=["s1_entity_id", "candidate_entity_id"], inplace=True)
    df_cand.reset_index(drop=True, inplace=True)

    print(f"Writing to {out_parquet.name}...")
    df_cand.to_parquet(out_parquet, index=False, engine="pyarrow", compression="snappy")
    print(f"Saved {len(df_cand):,} rows to {out_parquet} ({out_parquet.stat().st_size / 1e6:.1f} MB)")
    del df_cand
    gc.collect()


def find_norm_file(norm_dir: Path, src: str) -> Path:
    for name in [f"{src}_test_normalized.parquet", f"test_{src}_normalized.parquet"]:
        p = norm_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"Normalized file for {src} not found in {norm_dir}")


def main():
    parser = argparse.ArgumentParser(description="Task 7: Test Blocking V3")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip completed sources")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 80)
    print("STEP 2: TEST CANDIDATE GENERATION USING V3 BLOCKING (HIGH RECALL & LEAN)")
    print("=" * 80)

    # 1. Load S1 normalized table
    s1_norm_path = find_norm_file(NORM_DIR, "s1")
    print(f"Loading S1 test table from {s1_norm_path.name}...")
    df_s1 = pd.read_parquet(s1_norm_path)
    print(f"Loaded {len(df_s1):,} S1 entities.")

    s1_profiles = build_s1_v3_profiles(df_s1)

    # 2. Process Source 2
    s2_out = BLOCKING_DIR / "test_s1_s2_candidates.parquet"
    if args.resume and s2_out.is_file():
        print(f"\n[Source 2] Candidates already exist at {s2_out.name} (skipping)")
    else:
        print(f"\n[Source 2] Processing...")
        s2_norm_path = find_norm_file(NORM_DIR, "s2")
        s2_block_pairs = stream_v3_inverted_index(s1_profiles, s2_norm_path, "Source 2")
        s2_tfidf = run_partitioned_tfidf_blocking(df_s1, s2_norm_path, min_sim=0.45, top_k=15)
        s2_block_pairs["partitioned_tfidf"] = s2_tfidf
        combine_and_save_candidates(s2_block_pairs, s2_out)
        del s2_block_pairs, s2_tfidf
        gc.collect()

    # 3. Process Source 3
    s3_out = BLOCKING_DIR / "test_s1_s3_candidates.parquet"
    if args.resume and s3_out.is_file():
        print(f"\n[Source 3] Candidates already exist at {s3_out.name} (skipping)")
    else:
        print(f"\n[Source 3] Processing...")
        s3_norm_path = find_norm_file(NORM_DIR, "s3")
        s3_block_pairs = stream_v3_inverted_index(s1_profiles, s3_norm_path, "Source 3")
        s3_tfidf = run_partitioned_tfidf_blocking(df_s1, s3_norm_path, min_sim=0.45, top_k=15)
        s3_block_pairs["partitioned_tfidf"] = s3_tfidf
        combine_and_save_candidates(s3_block_pairs, s3_out)
        del s3_block_pairs, s3_tfidf
        gc.collect()

    print("\n" + "=" * 80)
    print(f"V3 TEST BLOCKING COMPLETED IN {time.time() - t_start:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
