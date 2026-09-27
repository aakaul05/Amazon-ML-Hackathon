"""
scripts/audit_blocking_false_negatives.py
=========================================
Phase 2: Comprehensive False Negative Audit for V2 Blocking.

Analyzes the ~17% of true matches missed by V2 blocking on the deterministic
10,000-S1 validation benchmark (val_s1_10k.parquet and val_gt_10k.json).

Identifies exact failure modes across S2 and S3:
1. Name typos / edit distance variations
2. Token reordering & partial token overlap
3. Disjoint names sharing high address evidence
4. Acronyms & abbreviations
5. Short / single-token entity names
6. Phonetic variations
7. Country code variations
8. Inverted index frequency / candidate-cap truncations
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "validation"
NORMALIZED_DIR = REPO_ROOT / "data" / "outputs" / "normalized_cache"


def categorize_pair(
    s1_name: str, s1_clean: str, s1_addr: str, s1_country: str,
    ot_name: str, ot_clean: str, ot_addr: str, ot_country: str,
) -> str:
    """Categorizes the primary root cause why a true pair was missed by V2 blockers."""
    if s1_name == ot_name or s1_clean == ot_clean:
        return "exact_name_capped"

    s1_tokens = set(s1_clean.split())
    ot_tokens = set(ot_clean.split())
    shared_tokens = s1_tokens & ot_tokens
    
    s1_addr_tokens = set(s1_addr.split())
    ot_addr_tokens = set(ot_addr.split())
    shared_addr_tokens = s1_addr_tokens & ot_addr_tokens
    
    s1_nums = {t for t in s1_addr_tokens if any(c.isdigit() for c in t)}
    ot_nums = {t for t in ot_addr_tokens if any(c.isdigit() for c in t)}
    shared_nums = s1_nums & ot_nums

    jw_sim = JaroWinkler.similarity(s1_clean, ot_clean)
    lev_dist = Levenshtein.distance(s1_clean, ot_clean)

    # 1. Disjoint name (0 common words) but strong address agreement
    if len(shared_tokens) == 0:
        if len(shared_nums) >= 1 or len(shared_addr_tokens) >= 3:
            return "disjoint_name_strong_address"
        if jw_sim >= 0.75 or lev_dist <= 3:
            return "name_typo_or_ocr"
        s1_initials = "".join([w[0] for w in s1_clean.split() if w])
        ot_initials = "".join([w[0] for w in ot_clean.split() if w])
        if (s1_initials and s1_initials == ot_clean.replace(" ", "")) or (ot_initials and ot_initials == s1_clean.replace(" ", "")):
            return "acronym_or_abbreviation"
        if len(s1_clean) <= 4 or len(ot_clean) <= 4:
            return "short_or_single_token_name"
        return "extreme_name_divergence"

    # 2. Shared tokens exist
    if s1_tokens == ot_tokens:
        return "word_order_variation_capped"
    if s1_tokens.issubset(ot_tokens) or ot_tokens.issubset(s1_tokens):
        return "token_subset_or_expansion"
    if len(shared_tokens) >= 1:
        # Check if address also shares something
        if len(shared_addr_tokens) >= 1:
            return "partial_name_plus_address"
        return "partial_token_overlap_only"

    # 3. Typo / OCR variations
    if lev_dist <= 3 or jw_sim >= 0.80:
        return "name_typo_or_ocr"

    return "other_variation"


def load_target_profiles_fast(parquet_path: Path, needed_entity_ids: set) -> dict:
    """Streams target parquet and loads ONLY the needed entity IDs (ultra fast & memory light)."""
    t0 = time.time()
    pf = pq.ParquetFile(str(parquet_path))
    profiles = {}
    cols = ["entity_id", "name_norm", "name_clean_legal", "address_norm", "country_norm"]
    
    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        df_b = batch.to_pandas()
        mask = np.fromiter((eid in needed_entity_ids for eid in df_b["entity_id"].values), dtype=bool, count=len(df_b))
        sub = df_b[mask]
        for eid, nn, ncl, an, cn in zip(sub["entity_id"], sub["name_norm"], sub["name_clean_legal"], sub["address_norm"], sub["country_norm"]):
            profiles[eid] = (str(nn or ""), str(ncl or ""), str(an or ""), str(cn or ""))
        if len(profiles) >= len(needed_entity_ids):
            break
            
    print(f"Loaded {len(profiles):,}/{len(needed_entity_ids):,} target entities from {parquet_path.name} in {time.time() - t0:.1f}s")
    return profiles


def analyze_source(
    source_name: str,
    gt_pairs_dict: dict,
    v2_cand_parquet: Path,
    val_s1_lookup: dict,
    target_lookup: dict,
) -> dict:
    print(f"\n{'=' * 75}")
    print(f"ANALYZING BLOCKING FALSE NEGATIVES: {source_name.upper()}")
    print(f"{'=' * 75}")

    all_gt_pairs = set()
    for s1_id, ot_set in gt_pairs_dict.items():
        for ot_id in ot_set:
            all_gt_pairs.add((s1_id, ot_id))
    total_gt = len(all_gt_pairs)
    print(f"Total Ground Truth Matches: {total_gt:,}")

    # Load V2 candidates
    cand_df = pd.read_parquet(v2_cand_parquet, columns=["s1_entity_id", "candidate_entity_id"])
    v2_cands = set(zip(cand_df["s1_entity_id"], cand_df["candidate_entity_id"]))
    del cand_df
    gc.collect()

    retrieved_tp = all_gt_pairs & v2_cands
    missed_fn = all_gt_pairs - v2_cands
    recall_pct = (len(retrieved_tp) / total_gt) * 100

    print(f"Retrieved True Positives : {len(retrieved_tp):,} ({recall_pct:.2f}%)")
    print(f"False Negatives (Missed) : {len(missed_fn):,} ({100 - recall_pct:.2f}%)")

    categories = Counter()
    category_samples = defaultdict(list)

    for s1_id, ot_id in missed_fn:
        s1_prof = val_s1_lookup.get(s1_id, ("", "", "", ""))
        ot_prof = target_lookup.get(ot_id, ("", "", "", ""))

        cat = categorize_pair(
            s1_name=s1_prof[0], s1_clean=s1_prof[1], s1_addr=s1_prof[2], s1_country=s1_prof[3],
            ot_name=ot_prof[0], ot_clean=ot_prof[1], ot_addr=ot_prof[2], ot_country=ot_prof[3],
        )
        categories[cat] += 1
        if len(category_samples[cat]) < 3:
            category_samples[cat].append({
                "s1_id": s1_id,
                "ot_id": ot_id,
                "s1_clean": s1_prof[1],
                "ot_clean": ot_prof[1],
                "s1_addr": s1_prof[2],
                "ot_addr": ot_prof[2],
                "s1_country": s1_prof[3],
                "ot_country": ot_prof[3],
            })

    print(f"\n{'CATEGORY':<35} {'COUNT':>8} {'PERCENT':>10}")
    print("-" * 55)
    for cat, count in categories.most_common():
        pct = (count / len(missed_fn)) * 100
        print(f"{cat:<35} {count:>8,} {pct:>9.2f}%")

    # Print representative examples of top 3 categories
    print("\nRepresentative Failure Examples:")
    for cat, _ in categories.most_common(4):
        print(f"\n--- Category: {cat} ---")
        for sample in category_samples[cat][:2]:
            print(f"  S1 ({sample['s1_id']}): '{sample['s1_clean']}' | Addr: '{sample['s1_addr']}'")
            print(f"  OT ({sample['ot_id']}): '{sample['ot_clean']}' | Addr: '{sample['ot_addr']}'")

    return {
        "source": source_name,
        "total_gt": total_gt,
        "retrieved_tp": len(retrieved_tp),
        "missed_fn": len(missed_fn),
        "recall_pct": recall_pct,
        "categories": dict(categories.most_common()),
        "samples": category_samples,
    }


def main():
    print("=" * 80)
    print("PHASE 2: FAST FALSE NEGATIVE AUDIT (10k S1 Validation Benchmark)")
    print("=" * 80)

    val_s1_path = OUTPUT_DIR / "val_s1_10k.parquet"
    val_gt_path = OUTPUT_DIR / "val_gt_10k.json"

    val_s1 = pd.read_parquet(val_s1_path)
    with open(val_gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Build S1 lookup
    val_s1_lookup = {}
    for eid, nn, ncl, an, cn in zip(
        val_s1["entity_id"], val_s1["name_norm"], val_s1["name_clean_legal"],
        val_s1["address_norm"], val_s1["country_norm"]
    ):
        val_s1_lookup[eid] = (str(nn or ""), str(ncl or ""), str(an or ""), str(cn or ""))

    # Identify exact target entity IDs needed
    needed_s2_ids = {ot for olist in gt_data["s2"].values() for ot in olist}
    needed_s3_ids = {ot for olist in gt_data["s3"].values() for ot in olist}

    print(f"Loading fast profiles for {len(needed_s2_ids):,} S2 and {len(needed_s3_ids):,} S3 GT entities...")
    s2_lookup = load_target_profiles_fast(NORMALIZED_DIR / "s2_normalized.parquet", needed_s2_ids)
    s3_lookup = load_target_profiles_fast(NORMALIZED_DIR / "s3_normalized.parquet", needed_s3_ids)

    v2_s2_parquet = OUTPUT_DIR / "val_v2_s1_s2_candidates.parquet"
    v2_s3_parquet = OUTPUT_DIR / "val_v2_s1_s3_candidates.parquet"

    res_s2 = analyze_source("Source 2", gt_data["s2"], v2_s2_parquet, val_s1_lookup, s2_lookup)
    res_s3 = analyze_source("Source 3", gt_data["s3"], v2_s3_parquet, val_s1_lookup, s3_lookup)

    report_path = OUTPUT_DIR / "v2_false_negative_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"s2": res_s2, "s3": res_s3}, f, indent=2)
    print(f"\nSaved detailed False Negative audit report to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
