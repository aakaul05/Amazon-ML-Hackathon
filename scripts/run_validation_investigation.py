"""
scripts/run_validation_investigation.py
=======================================
Phase 5 & Phase 6 Investigation on 10k Validation Set:
1. Runs all 10 blocking passes on the 10k validation set against Source 2.
2. Computes the Phase 6 Blocking Ablation Table:
   - Raw candidates
   - Unique candidates
   - GT recovered
   - Marginal (new) GT recovered
   - Marginal recall
   - Candidate / GT ratio
3. Identifies all missed GT pairs and performs Phase 5 Error Classification:
   - Classifies missed pairs into the 17 specified failure modes.
   - Outputs counts and percentages.
"""

from pathlib import Path
import os
import sys
import json
import time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.blocking import (
    block_exact_field,
    block_rare_tokens,
    block_address_tokens,
    block_tfidf_char_ngram,
    block_sorted_tokens,
    block_name_prefix,
    block_country_name_token,
    block_phonetic,
)
from business_entity_resolution.blocking.candidate_generator import _double_metaphone_simple

OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NORMALIZED_DIR = REPO_ROOT / "data" / "outputs" / "normalized_cache"

def main():
    t_start = time.time()
    print("=" * 75)
    print("PHASE 5 & 6 INVESTIGATION: 10K VALIDATION BENCHMARK (S1 -> S2)")
    print("=" * 75)

    # 1. Load 10k S1 validation set and ground truth
    val_s1_path = OUTPUT_DIR / "val_s1_10k.parquet"
    if not val_s1_path.exists():
        raise FileNotFoundError(f"{val_s1_path} not found. Run create_validation_benchmark.py first.")
    
    val_s1 = pd.read_parquet(val_s1_path)
    print(f"Loaded {len(val_s1):,} validation S1 entities.")

    with open(OUTPUT_DIR / "val_gt_10k.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    
    gt_s2_dict = {k: set(v) for k, v in gt_data["s2"].items()}
    total_true_s2 = gt_data["stats"]["total_s2_pairs"]
    print(f"Loaded {total_true_s2:,} ground-truth pairs for S2.")

    gt_pairs_set = {(s1, s2) for s1, s2_list in gt_s2_dict.items() for s2 in s2_list}

    # 2. Load S2 normalized
    print("\nLoading Source 2 normalized cache...")
    cols = ["entity_id", "business_name", "business_address", "country", "name_norm", "name_clean_legal", "address_norm", "country_norm"]
    s2 = pd.read_parquet(NORMALIZED_DIR / "s2_normalized.parquet", columns=cols)
    print(f"Loaded {len(s2):,} Source 2 rows.")

    # 3. Run each blocking pass individually and record candidate pairs
    passes = [
        ("1. Exact name_norm", lambda: block_exact_field(val_s1, s2, "name_norm", "exact_name_norm")),
        ("2. Exact clean legal", lambda: block_exact_field(val_s1, s2, "name_clean_legal", "exact_clean_legal")),
        ("3. Rare tokens (DF<=500)", lambda: block_rare_tokens(val_s1, s2, max_df=500, max_cand_per_s1=40)),
        ("4. Address tokens", lambda: block_address_tokens(val_s1, s2)),
        ("5. TF-IDF char 3-gram (0.70)", lambda: block_tfidf_char_ngram(val_s1, s2, min_sim=0.70, top_k=10, sample_limit=len(s2))),
        ("6. TF-IDF char 3-gram (0.50)", lambda: block_tfidf_char_ngram(val_s1, s2, min_sim=0.50, top_k=10, sample_limit=len(s2))),
        ("7. Sorted-token blocking", lambda: block_sorted_tokens(val_s1, s2, max_key_df=300, max_cand_per_s1=40)),
        ("8. Name prefix (6 chars)", lambda: block_name_prefix(val_s1, s2, prefix_len=6, max_key_df=150, max_cand_per_s1=25)),
        ("9. Country + name token", lambda: block_country_name_token(val_s1, s2, max_key_df=250, max_cand_per_s1=30, min_token_len=4)),
        ("10. Phonetic blocking", lambda: block_phonetic(val_s1, s2, max_key_df=200, max_cand_per_s1=30)),
    ]

    pass_results = []
    cumulative_pairs = set()
    cumulative_tp = set()

    print("\n" + "=" * 75)
    print("RUNNING 10 BLOCKING PASSES & COMPUTING ABLATION")
    print("=" * 75)

    for pname, pfunc in passes:
        t0 = time.time()
        bdf = pfunc()
        elapsed = time.time() - t0
        
        # Extract pairs
        pairs = set(zip(bdf["s1_entity_id"], bdf["candidate_entity_id"]))
        tp_found = pairs & gt_pairs_set
        
        new_tp = tp_found - cumulative_tp
        cumulative_tp.update(tp_found)
        cumulative_pairs.update(pairs)

        pass_results.append({
            "blocker": pname,
            "raw_candidates": len(bdf),
            "unique_candidates": len(pairs),
            "gt_recovered": len(tp_found),
            "individual_recall": len(tp_found) / total_true_s2 if total_true_s2 > 0 else 0,
            "new_gt_recovered": len(new_tp),
            "cumulative_gt": len(cumulative_tp),
            "cumulative_recall": len(cumulative_tp) / total_true_s2 if total_true_s2 > 0 else 0,
            "cumulative_candidates": len(cumulative_pairs),
            "cand_cost_per_new_gt": len(pairs) / max(1, len(new_tp)),
            "elapsed_s": elapsed,
            "df": bdf,
        })
        print(f"Done: {pname} | Unique: {len(pairs):,} | TP: {len(tp_found):,} | New TP: {len(new_tp):,} ({elapsed:.1f}s)")

    # 4. Print Phase 6 Ablation Table
    print("\n" + "=" * 105)
    print("PHASE 6: BLOCKING ABLATION TABLE (S1 -> S2 on 10k Validation Set)")
    print("=" * 105)
    header = f"{'BLOCKER':<30} {'CANDIDATES':>12} {'GT RECOVERED':>14} {'NEW GT':>10} {'MARGINAL REC':>14} {'CUMULATIVE REC':>16} {'CAND/NEW_GT':>13}"
    print(header)
    print("-" * 105)
    for r in pass_results:
        marginal_rec = (r['new_gt_recovered'] / total_true_s2) * 100
        cum_rec = r['cumulative_recall'] * 100
        print(f"{r['blocker']:<30} {r['unique_candidates']:>12,} {r['gt_recovered']:>14,} {r['new_gt_recovered']:>10,} {marginal_rec:>13.2f}% {cum_rec:>15.2f}% {r['cand_cost_per_new_gt']:>13.1f}")
    print("-" * 105)
    print(f"{'TOTAL UNION':<30} {len(cumulative_pairs):>12,} {len(cumulative_tp):>14,} {len(cumulative_tp):>10,} {'-':>14} {len(cumulative_tp)/total_true_s2*100:>15.2f}% {len(cumulative_pairs)/max(1, len(cumulative_tp)):>13.1f}")
    print("=" * 105)

    # 5. Phase 5: Error Analysis on Missed GT Pairs
    missed_gt_pairs = gt_pairs_set - cumulative_tp
    print(f"\n" + "=" * 75)
    print(f"PHASE 5: ROOT-CAUSE ANALYSIS OF MISSED MATCHES")
    print(f"Total True Pairs: {total_true_s2:,} | Recovered: {len(cumulative_tp):,} ({len(cumulative_tp)/total_true_s2*100:.2f}%) | Missed: {len(missed_gt_pairs):,} ({len(missed_gt_pairs)/total_true_s2*100:.2f}%)")
    print("=" * 75)

    # Build lookup for S1 and S2 records
    s1_lookup = val_s1.set_index("entity_id").to_dict(orient="index")
    
    # We only need S2 records that are in missed_gt_pairs
    missed_s2_ids = {s2_id for _, s2_id in missed_gt_pairs}
    s2_missed_sub = s2[s2["entity_id"].isin(missed_s2_ids)].set_index("entity_id").to_dict(orient="index")

    # Classify each missed pair into the 17 categories
    # 1. spelling variation
    # 2. abbreviation
    # 3. token reorder
    # 4. punctuation
    # 5. legal suffix
    # 6. transliteration
    # 7. missing name token
    # 8. address variation
    # 9. address abbreviation
    # 10. numeric/address variation
    # 11. country mismatch
    # 12. common name
    # 13. rare-token miss
    # 14. TF-IDF miss
    # 15. candidate-cap truncation
    # 16. phonetic miss
    # 17. other

    failure_counts = Counter()
    failure_examples = defaultdict(list)

    LEGAL_WORDS = {"inc", "corp", "corporation", "incorporated", "llc", "ltd", "limited", "pvt", "private", "co", "company", "plc", "llp", "gmbh", "ag", "sa", "srl", "sl"}

    for s1_id, s2_id in missed_gt_pairs:
        r1 = s1_lookup.get(s1_id)
        r2 = s2_missed_sub.get(s2_id)
        if not r1 or not r2:
            continue

        n1 = r1["name_clean_legal"]
        n2 = r2["name_clean_legal"]
        raw1 = r1["name_norm"]
        raw2 = r2["name_norm"]
        a1 = r1["address_norm"]
        a2 = r2["address_norm"]
        c1 = r1["country_norm"]
        c2 = r2["country_norm"]

        t1 = set(n1.split())
        t2 = set(n2.split())

        # Determine primary failure reason
        assigned = False

        # Country mismatch
        if c1 != c2:
            failure_counts["11. country mismatch"] += 1
            if len(failure_examples["11. country mismatch"]) < 3:
                failure_examples["11. country mismatch"].append((s1_id, s2_id, n1, n2, c1, c2))
            assigned = True

        # Legal suffix failure (unstripped or different)
        elif raw1 != raw2 and set(raw1.split()) - set(raw2.split()) <= LEGAL_WORDS and set(raw2.split()) - set(raw1.split()) <= LEGAL_WORDS:
            failure_counts["5. legal suffix"] += 1
            if len(failure_examples["5. legal suffix"]) < 3:
                failure_examples["5. legal suffix"].append((s1_id, s2_id, raw1, raw2))
            assigned = True

        # Token reorder (exact same tokens, different order)
        elif t1 and t1 == t2 and n1 != n2:
            failure_counts["3. token reorder"] += 1
            if len(failure_examples["3. token reorder"]) < 3:
                failure_examples["3. token reorder"].append((s1_id, s2_id, n1, n2))
            assigned = True

        # Punctuation / whitespace artifact
        elif raw1.replace(" ", "") == raw2.replace(" ", ""):
            failure_counts["4. punctuation"] += 1
            if len(failure_examples["4. punctuation"]) < 3:
                failure_examples["4. punctuation"].append((s1_id, s2_id, raw1, raw2))
            assigned = True

        # Abbreviation (e.g. Intl vs International, or initials)
        elif any(len(tok) <= 4 and any(tok in other_tok and len(other_tok) > len(tok) + 2 for other_tok in t2) for tok in t1) or \
             any(len(tok) <= 4 and any(tok in other_tok and len(other_tok) > len(tok) + 2 for other_tok in t1) for tok in t2):
            failure_counts["2. abbreviation"] += 1
            if len(failure_examples["2. abbreviation"]) < 3:
                failure_examples["2. abbreviation"].append((s1_id, s2_id, n1, n2))
            assigned = True

        # Spelling variation (high edit similarity, small distance)
        elif n1 and n2 and (Levenshtein.distance(n1, n2) <= 2 or fuzz.ratio(n1, n2) >= 80):
            failure_counts["1. spelling variation"] += 1
            if len(failure_examples["1. spelling variation"]) < 3:
                failure_examples["1. spelling variation"].append((s1_id, s2_id, n1, n2))
            assigned = True

        # Phonetic equivalence
        elif n1 and n2 and _double_metaphone_simple(n1.split()[0]) == _double_metaphone_simple(n2.split()[0]):
            failure_counts["16. phonetic miss"] += 1
            if len(failure_examples["16. phonetic miss"]) < 3:
                failure_examples["16. phonetic miss"].append((s1_id, s2_id, n1, n2))
            assigned = True

        # Missing name token (one is subset of the other or extra descriptive word)
        elif (t1 and t1.issubset(t2)) or (t2 and t2.issubset(t1)) or (len(t1 & t2) >= 1):
            failure_counts["7. missing name token"] += 1
            if len(failure_examples["7. missing name token"]) < 3:
                failure_examples["7. missing name token"].append((s1_id, s2_id, n1, n2))
            assigned = True

        # Transliteration
        elif fuzz.partial_ratio(n1, n2) >= 70:
            failure_counts["6. transliteration"] += 1
            if len(failure_examples["6. transliteration"]) < 3:
                failure_examples["6. transliteration"].append((s1_id, s2_id, n1, n2))
            assigned = True

        # Address variation / numeric
        elif a1 and a2 and any(c.isdigit() for c in a1) and any(c.isdigit() for c in a2):
            failure_counts["10. numeric/address variation"] += 1
            if len(failure_examples["10. numeric/address variation"]) < 3:
                failure_examples["10. numeric/address variation"].append((s1_id, s2_id, a1, a2))
            assigned = True

        # Common name / low distinctiveness
        elif len(t1 & t2) == 0 and fuzz.ratio(n1, n2) < 40:
            failure_counts["12. common name"] += 1
            if len(failure_examples["12. common name"]) < 3:
                failure_examples["12. common name"].append((s1_id, s2_id, n1, n2))
            assigned = True

        else:
            failure_counts["17. other"] += 1
            if len(failure_examples["17. other"]) < 3:
                failure_examples["17. other"].append((s1_id, s2_id, n1, n2, a1, a2))

    print(f"\n{'FAILURE MODE':<35} {'COUNT':>8} {'PERCENT':>10}")
    print("-" * 55)
    total_classified = sum(failure_counts.values())
    for fmode, cnt in failure_counts.most_common():
        pct = (cnt / total_classified) * 100 if total_classified > 0 else 0
        print(f"{fmode:<35} {cnt:>8,} {pct:>9.2f}%")
    print("-" * 55)
    print(f"{'TOTAL MISSED':<35} {total_classified:>8,} {100.0:>9.2f}%")
    print("=" * 75)

    # Save summary report to JSON
    report = {
        "benchmark": "10k S1 -> S2 Validation",
        "total_true_pairs": total_true_s2,
        "recovered_tp": len(cumulative_tp),
        "union_recall": len(cumulative_tp) / total_true_s2,
        "total_candidates": len(cumulative_pairs),
        "candidates_per_s1": len(cumulative_pairs) / len(val_s1),
        "ablation": [
            {
                "blocker": r["blocker"],
                "unique_candidates": r["unique_candidates"],
                "gt_recovered": r["gt_recovered"],
                "new_gt_recovered": r["new_gt_recovered"],
                "cand_cost_per_new_gt": r["cand_cost_per_new_gt"],
            }
            for r in pass_results
        ],
        "failure_modes": {k: v for k, v in failure_counts.most_common()},
    }
    with open(OUTPUT_DIR / "validation_investigation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved investigation report: {OUTPUT_DIR / 'validation_investigation_report.json'}")

if __name__ == "__main__":
    main()
