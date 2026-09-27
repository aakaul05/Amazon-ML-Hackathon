"""
scripts/13_v3_blocking_benchmark.py
====================================
V3 Blocking Benchmark — Comprehensive Analysis & Improvement

Phases:
  1. Create deterministic 10k validation benchmark (SEED=42)
  2. Run all V1 passes individually with per-pass GT recall
  3. Analyze false negatives: categorize WHY each GT pair is missed
  4. Design & test new targeted blocking strategies
  5. Ablation / combination search
  6. Final report

IMPORTANT:
  - Does NOT overwrite any V1/V2 artifacts
  - Does NOT run full test or CatBoost
  - All outputs go to data/student_resource/outputs/v3_blocking/
"""

import gc
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ── Setup ─────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RANDOM_SEED = 42
VAL_S1_SIZE = 10000

TRAIN_DIR = REPO_ROOT / "data" / "student_resource" / "dataset" / "train"
V3_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "v3_blocking"
V3_DIR.mkdir(parents=True, exist_ok=True)

from business_entity_resolution.preprocessing.normalization import (
    normalize_series_fast,
    load_normalized_or_compute,
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
    _double_metaphone_simple,
)


# ================================================================
# PHASE 1: CREATE DETERMINISTIC VALIDATION BENCHMARK
# ================================================================

def parse_ground_truth() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], int, int]:
    """Parse ground truth into S2 and S3 dicts."""
    gt_s2 = defaultdict(set)
    gt_s3 = defaultdict(set)
    total_s2 = 0
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
                if m.startswith("S2-"):
                    gt_s2[s1_id].add(m)
                    total_s2 += 1
                elif m.startswith("S3-"):
                    gt_s3[s1_id].add(m)
                    total_s3 += 1

    return gt_s2, gt_s3, total_s2, total_s3


def create_validation_benchmark():
    """Create deterministic 10k S1 sample and ground truth subset."""
    print("\n" + "=" * 70)
    print("PHASE 1: CREATING DETERMINISTIC VALIDATION BENCHMARK")
    print("=" * 70)

    # Load normalized data
    cols = ["entity_id", "business_name", "name_norm", "name_clean_legal",
            "address_norm", "country_norm"]
    s1, s2, s3 = load_normalized_or_compute(TRAIN_DIR, REPO_ROOT, columns=cols)

    print(f"Full S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}")

    # Parse full ground truth
    gt_s2_full, gt_s3_full, total_s2, total_s3 = parse_ground_truth()
    print(f"Total GT pairs — S2: {total_s2:,}  S3: {total_s3:,}")

    # Deterministic S1 sample
    rng = np.random.RandomState(RANDOM_SEED)
    s1_ids = s1["entity_id"].values
    sample_idx = rng.choice(len(s1_ids), size=VAL_S1_SIZE, replace=False)
    sample_idx.sort()
    val_s1_ids = set(s1_ids[sample_idx])

    val_s1 = s1[s1["entity_id"].isin(val_s1_ids)].copy().reset_index(drop=True)

    # Ground truth for sampled S1
    val_gt_s2 = {s1_id: gt_s2_full[s1_id] for s1_id in val_s1_ids if s1_id in gt_s2_full}
    val_gt_s3 = {s1_id: gt_s3_full[s1_id] for s1_id in val_s1_ids if s1_id in gt_s3_full}
    val_total_s2 = sum(len(v) for v in val_gt_s2.values())
    val_total_s3 = sum(len(v) for v in val_gt_s3.values())

    print(f"\nValidation S1 sample: {len(val_s1):,}")
    print(f"Val GT S2 pairs: {val_total_s2:,} (from {len(val_gt_s2):,} S1 entities)")
    print(f"Val GT S3 pairs: {val_total_s3:,} (from {len(val_gt_s3):,} S1 entities)")

    # Save artifacts
    val_s1.to_parquet(V3_DIR / "val_s1_10k.parquet", index=False)

    # Save GT as JSON (convert sets to lists)
    gt_json = {
        "s1_s2": {k: sorted(list(v)) for k, v in val_gt_s2.items()},
        "s1_s3": {k: sorted(list(v)) for k, v in val_gt_s3.items()},
    }
    with open(V3_DIR / "val_gt_10k.json", "w") as f:
        json.dump(gt_json, f)

    return val_s1, s2, s3, val_gt_s2, val_gt_s3, val_total_s2, val_total_s3


def load_validation_benchmark():
    """Load existing or create validation benchmark."""
    val_s1_path = V3_DIR / "val_s1_10k.parquet"
    val_gt_path = V3_DIR / "val_gt_10k.json"

    if val_s1_path.exists() and val_gt_path.exists():
        print("Loading existing V3 validation benchmark...")
        val_s1 = pd.read_parquet(val_s1_path)
        with open(val_gt_path) as f:
            gt_json = json.load(f)
        val_gt_s2 = {k: set(v) for k, v in gt_json["s1_s2"].items()}
        val_gt_s3 = {k: set(v) for k, v in gt_json["s1_s3"].items()}
        val_total_s2 = sum(len(v) for v in val_gt_s2.values())
        val_total_s3 = sum(len(v) for v in val_gt_s3.values())

        # Load full S2, S3
        cols = ["entity_id", "business_name", "name_norm", "name_clean_legal",
                "address_norm", "country_norm"]
        _, s2, s3 = load_normalized_or_compute(TRAIN_DIR, REPO_ROOT, columns=cols)
        return val_s1, s2, s3, val_gt_s2, val_gt_s3, val_total_s2, val_total_s3
    else:
        return create_validation_benchmark()


# ================================================================
# PHASE 2: RUN ALL V1 PASSES WITH PER-PASS METRICS
# ================================================================

def evaluate_single_block(block_df, gt_dict, total_gt, label=""):
    """Evaluate a single blocking pass against ground truth."""
    if block_df.empty:
        return {"name": label, "candidates": 0, "gt_found": 0, "recall": 0.0}

    pairs = block_df[["s1_entity_id", "candidate_entity_id"]].drop_duplicates()
    # Count GT pairs found
    gt_found = 0
    for s1_id, ot_id in zip(pairs["s1_entity_id"], pairs["candidate_entity_id"]):
        if s1_id in gt_dict and ot_id in gt_dict[s1_id]:
            gt_found += 1

    recall = gt_found / total_gt if total_gt > 0 else 0.0
    return {
        "name": label,
        "candidates": len(pairs),
        "gt_found": gt_found,
        "recall": recall,
    }


def evaluate_incremental(block_dfs, block_names, gt_dict, total_gt):
    """Evaluate passes incrementally to see marginal GT recovery."""
    all_found = set()
    results = []

    for bdf, bname in zip(block_dfs, block_names):
        if bdf.empty:
            results.append({
                "pass": bname, "candidates": 0, "gt_found": 0,
                "new_gt": 0, "cumulative_gt": len(all_found),
                "incremental_recall": 0.0, "cumulative_recall": len(all_found) / total_gt if total_gt > 0 else 0.0,
            })
            continue

        pairs = bdf[["s1_entity_id", "candidate_entity_id"]].drop_duplicates()
        pass_found = set()
        for s1_id, ot_id in zip(pairs["s1_entity_id"], pairs["candidate_entity_id"]):
            if s1_id in gt_dict and ot_id in gt_dict[s1_id]:
                pass_found.add((s1_id, ot_id))

        new_gt = pass_found - all_found
        all_found.update(pass_found)

        results.append({
            "pass": bname,
            "candidates": len(pairs),
            "gt_found": len(pass_found),
            "new_gt": len(new_gt),
            "cumulative_gt": len(all_found),
            "incremental_recall": len(new_gt) / total_gt if total_gt > 0 else 0.0,
            "cumulative_recall": len(all_found) / total_gt if total_gt > 0 else 0.0,
        })

    return results, all_found


def run_v1_passes(val_s1, other_df, source_label):
    """Run all 10 V1 blocking passes with V1 parameters."""
    print(f"\n{'#' * 70}")
    print(f"# V1 BLOCKING: S1 → {source_label}")
    print(f"# S1: {len(val_s1):,} rows  |  {source_label}: {len(other_df):,} rows")
    print(f"{'#' * 70}")

    blocks = []
    names = []

    def _run(name, fn):
        t0 = time.time()
        print(f"\n--- {name} ---", flush=True)
        try:
            res = fn()
            print(f"    -> {len(res):,} pairs ({time.time() - t0:.1f}s)", flush=True)
            gc.collect()
            return res
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
            return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "block"])

    # V1 parameters (from 07_improved_blocking.py)
    names.append("1_exact_name_norm")
    blocks.append(_run("Pass 1: exact_name_norm",
        lambda: block_exact_field(val_s1, other_df, "name_norm", "exact_name_norm")))

    names.append("2_exact_clean_legal")
    blocks.append(_run("Pass 2: exact_clean_legal",
        lambda: block_exact_field(val_s1, other_df, "name_clean_legal", "exact_clean_legal")))

    names.append("3_rare_token")
    blocks.append(_run("Pass 3: rare_token (DF<=500)",
        lambda: block_rare_tokens(val_s1, other_df, max_df=500, max_cand_per_s1=40)))

    names.append("4_address_component")
    blocks.append(_run("Pass 4: address_component",
        lambda: block_address_tokens(val_s1, other_df)))

    names.append("5_char_ngram_0.70")
    blocks.append(_run("Pass 5: char_ngram (sim>=0.70, top_k=10)",
        lambda: block_tfidf_char_ngram(val_s1, other_df, min_sim=0.70, top_k=10, sample_limit=300000)))

    names.append("6_char_ngram_0.50")
    blocks.append(_run("Pass 6: char_ngram (sim>=0.50, top_k=10)",
        lambda: block_tfidf_char_ngram(val_s1, other_df, min_sim=0.50, top_k=10, sample_limit=300000)))

    names.append("7_sorted_token")
    blocks.append(_run("Pass 7: sorted_token",
        lambda: block_sorted_tokens(val_s1, other_df, max_key_df=300, max_cand_per_s1=40)))

    names.append("8_name_prefix_6")
    blocks.append(_run("Pass 8: name_prefix_6",
        lambda: block_name_prefix(val_s1, other_df, prefix_len=6, max_key_df=150, max_cand_per_s1=25)))

    names.append("9_country_name_token")
    blocks.append(_run("Pass 9: country_name_token",
        lambda: block_country_name_token(val_s1, other_df, max_key_df=250, max_cand_per_s1=30, min_token_len=4)))

    names.append("10_phonetic")
    blocks.append(_run("Pass 10: phonetic",
        lambda: block_phonetic(val_s1, other_df, max_key_df=200, max_cand_per_s1=30)))

    return blocks, names


# ================================================================
# PHASE 3: FALSE NEGATIVE ANALYSIS
# ================================================================

def analyze_false_negatives(val_s1, other_df, gt_dict, total_gt,
                            found_pairs, source_label):
    """Analyze why each missed GT pair was not found by blocking."""
    print(f"\n{'=' * 70}")
    print(f"PHASE 3: FALSE NEGATIVE ANALYSIS — {source_label}")
    print(f"{'=' * 70}")

    # Build full GT pair set
    all_gt_pairs = set()
    for s1_id, matches in gt_dict.items():
        for ot_id in matches:
            all_gt_pairs.add((s1_id, ot_id))

    missed_pairs = all_gt_pairs - found_pairs
    print(f"Total GT pairs: {len(all_gt_pairs):,}")
    print(f"Found pairs:    {len(found_pairs):,}")
    print(f"MISSED pairs:   {len(missed_pairs):,}")
    print(f"Miss rate:      {len(missed_pairs)/len(all_gt_pairs)*100:.2f}%")

    if not missed_pairs:
        return {}

    # Build lookup dicts for fast access
    s1_lookup = val_s1.set_index("entity_id").to_dict("index")
    other_lookup = other_df.set_index("entity_id").to_dict("index")

    # Categorize missed pairs
    categories = defaultdict(list)

    for s1_id, ot_id in sorted(missed_pairs):
        s1_rec = s1_lookup.get(s1_id, {})
        ot_rec = other_lookup.get(ot_id, {})

        if not s1_rec or not ot_rec:
            categories["Z_LOOKUP_FAIL"].append((s1_id, ot_id, {}, {}))
            continue

        s1_name = s1_rec.get("name_norm", "")
        s1_clean = s1_rec.get("name_clean_legal", "")
        s1_addr = s1_rec.get("address_norm", "")
        s1_country = s1_rec.get("country_norm", "")
        s1_bname = s1_rec.get("business_name", "")

        ot_name = ot_rec.get("name_norm", "")
        ot_clean = ot_rec.get("name_clean_legal", "")
        ot_addr = ot_rec.get("address_norm", "")
        ot_country = ot_rec.get("country_norm", "")
        ot_bname = ot_rec.get("business_name", "")

        reasons = []

        # Check name token overlap
        s1_tokens = set(s1_clean.split()) if s1_clean else set()
        ot_tokens = set(ot_clean.split()) if ot_clean else set()
        common_tokens = s1_tokens & ot_tokens
        all_tokens = s1_tokens | ot_tokens

        if not s1_tokens or not ot_tokens:
            jaccard = 0.0
        else:
            jaccard = len(common_tokens) / len(all_tokens)

        # Check prefix overlap
        prefix_match = (s1_clean[:6] == ot_clean[:6]) if s1_clean and ot_clean and len(s1_clean) >= 6 and len(ot_clean) >= 6 else False

        # Check country
        country_match = s1_country == ot_country if s1_country and ot_country else False
        country_mismatch = (s1_country != ot_country) and s1_country and ot_country

        # Check address token overlap
        s1_addr_tokens = set(s1_addr.split()) if s1_addr else set()
        ot_addr_tokens = set(ot_addr.split()) if ot_addr else set()
        addr_common = s1_addr_tokens & ot_addr_tokens

        # Extract numeric tokens from addresses
        s1_nums = {t for t in s1_addr.split() if any(c.isdigit() for c in t) and len(t) >= 2} if s1_addr else set()
        ot_nums = {t for t in ot_addr.split() if any(c.isdigit() for c in t) and len(t) >= 2} if ot_addr else set()
        num_overlap = s1_nums & ot_nums

        # Phonetic keys
        def phon_key(name):
            toks = name.split()[:2] if name else []
            codes = [_double_metaphone_simple(t) for t in toks]
            return "|".join(c for c in codes if c)

        s1_phon = phon_key(s1_clean)
        ot_phon = phon_key(ot_clean)
        phon_match = s1_phon == ot_phon if s1_phon and ot_phon else False

        # Sorted token key
        s1_sorted = " ".join(sorted(s1_tokens)) if len(s1_tokens) >= 2 else ""
        ot_sorted = " ".join(sorted(ot_tokens)) if len(ot_tokens) >= 2 else ""
        sorted_match = s1_sorted == ot_sorted if s1_sorted and ot_sorted else False

        # ── Categorize ──
        info = {
            "s1_id": s1_id, "ot_id": ot_id,
            "s1_name": s1_bname, "s1_clean": s1_clean, "s1_addr": s1_addr, "s1_country": s1_country,
            "ot_name": ot_bname, "ot_clean": ot_clean, "ot_addr": ot_addr, "ot_country": ot_country,
            "jaccard": jaccard, "common_tokens": common_tokens,
            "prefix_match": prefix_match, "country_match": country_match,
            "phon_match": phon_match, "sorted_match": sorted_match,
            "addr_num_overlap": num_overlap,
        }

        if jaccard == 0.0 and not prefix_match:
            # No token overlap at all — likely acronym, abbreviation, or totally different name
            if len(s1_clean) <= 4 or len(ot_clean) <= 4:
                categories["K_SHORT_NAME"].append(info)
            elif any(len(t) <= 2 for t in s1_tokens) or any(len(t) <= 2 for t in ot_tokens):
                categories["L_ACRONYM"].append(info)
            else:
                # Check if it could be transliteration
                # Compute character-level similarity
                s1_chars = set(s1_clean.replace(" ", ""))
                ot_chars = set(ot_clean.replace(" ", ""))
                char_jaccard = len(s1_chars & ot_chars) / len(s1_chars | ot_chars) if (s1_chars | ot_chars) else 0.0
                if char_jaccard >= 0.3:
                    categories["E_TRANSLITERATION"].append(info)
                else:
                    categories["O_NO_OVERLAP"].append(info)

        elif 0.0 < jaccard <= 0.3:
            # Very low overlap — usually one meaningful token shared
            if country_mismatch:
                categories["J_COUNTRY_VARIATION"].append(info)
            elif len(common_tokens) == 1 and len(all_tokens) >= 5:
                categories["N_MULTI_TOKEN_VARIATION"].append(info)
            else:
                categories["A_NAME_VARIATION"].append(info)

        elif 0.3 < jaccard < 1.0:
            # Partial overlap — identify what differs
            s1_only = s1_tokens - ot_tokens
            ot_only = ot_tokens - s1_tokens

            # Check if differences are legal suffixes
            legal_terms = {"inc", "corp", "corporation", "limited", "ltd", "llc", "llp",
                           "plc", "gmbh", "ag", "sa", "srl", "spa", "bv", "nv", "pvt",
                           "co", "company", "ab", "as", "oy", "sas", "kk", "pty"}
            s1_only_no_legal = s1_only - legal_terms
            ot_only_no_legal = ot_only - legal_terms

            if not s1_only_no_legal and not ot_only_no_legal:
                categories["B_LEGAL_SUFFIX"].append(info)
            elif sorted_match:
                categories["C_TOKEN_ORDER"].append(info)
            else:
                # Check for typos — edit distance proxy
                has_typo = False
                for t1 in s1_only_no_legal:
                    for t2 in ot_only_no_legal:
                        if abs(len(t1) - len(t2)) <= 2:
                            # Quick Levenshtein approximation
                            common_chars = sum(1 for a, b in zip(t1, t2) if a == b)
                            if common_chars >= max(len(t1), len(t2)) * 0.6:
                                has_typo = True
                                break
                    if has_typo:
                        break

                if has_typo:
                    categories["D_TYPO_SPELLING"].append(info)
                elif country_mismatch:
                    categories["J_COUNTRY_VARIATION"].append(info)
                elif not s1_addr or not ot_addr:
                    categories["G_ADDRESS_MISSING"].append(info)
                else:
                    categories["A_NAME_VARIATION"].append(info)

        elif jaccard == 1.0:
            # Same tokens — but still missed? Should have been caught by exact/sorted
            # Probably a frequency cap issue
            categories["P_FREQ_CAP_ISSUE"].append(info)

        else:
            categories["O_OTHER"].append(info)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"FALSE NEGATIVE CATEGORY SUMMARY — {source_label}")
    print(f"{'=' * 70}")
    print(f"{'Category':<35} {'Count':>6} {'%':>7}")
    print("-" * 50)
    for cat in sorted(categories.keys()):
        cnt = len(categories[cat])
        pct = cnt / len(missed_pairs) * 100
        print(f"{cat:<35} {cnt:>6} {pct:>6.1f}%")
    print("-" * 50)
    print(f"{'TOTAL MISSED':<35} {len(missed_pairs):>6}")

    # Print examples for each category (up to 5 per category)
    print(f"\n{'=' * 70}")
    print(f"REPRESENTATIVE EXAMPLES — {source_label}")
    print(f"{'=' * 70}")
    for cat in sorted(categories.keys()):
        items = categories[cat]
        print(f"\n--- {cat} ({len(items)} pairs) ---")
        for info in items[:5]:
            print(f"  S1 [{info['s1_id']}]:")
            print(f"    name_clean = {info['s1_clean']}")
            print(f"    address    = {info['s1_addr']}")
            print(f"    country    = {info['s1_country']}")
            print(f"  GT [{info['ot_id']}]:")
            print(f"    name_clean = {info['ot_clean']}")
            print(f"    address    = {info['ot_addr']}")
            print(f"    country    = {info['ot_country']}")
            print(f"    jaccard    = {info['jaccard']:.3f}")
            print(f"    common_tokens = {info['common_tokens']}")
            print(f"    prefix_match = {info['prefix_match']}")
            print(f"    phon_match = {info['phon_match']}")
            print()

    # Save full analysis
    analysis = {cat: len(items) for cat, items in categories.items()}
    analysis["total_missed"] = len(missed_pairs)
    analysis["total_gt"] = len(all_gt_pairs)
    analysis["found"] = len(found_pairs)

    return categories


# ================================================================
# PHASE 4: NEW BLOCKING STRATEGIES
# ================================================================

def block_first_informative_token(s1_df, other_df, max_df=300, max_cand_per_s1=50):
    """Block on the first informative (non-stopword, len>=3) token of name_clean_legal."""
    t0 = time.time()
    block_name = "first_info_token"

    STOP_TOKENS = {
        "the", "and", "for", "of", "in", "de", "del", "la", "el", "le", "les",
        "das", "der", "die", "van", "von", "den", "het",
        "group", "services", "company", "international", "global", "solutions",
        "management", "systems", "technologies", "consulting", "enterprises",
        "partners", "associates", "holdings", "industries", "products", "national",
        "general", "american", "business", "financial", "capital", "investments",
    }

    def first_info_token(name):
        if not name:
            return ""
        for t in name.split():
            if len(t) >= 3 and t not in STOP_TOKENS:
                return t
        return ""

    # Build inverted index
    token_counts = Counter()
    other_valid = other_df[other_df["name_clean_legal"] != ""]
    for name in other_valid["name_clean_legal"]:
        tok = first_info_token(name)
        if tok:
            token_counts[tok] += 1

    valid_tokens = {t for t, c in token_counts.items() if c <= max_df}

    inverted = defaultdict(list)
    for eid, name in zip(other_valid["entity_id"], other_valid["name_clean_legal"]):
        tok = first_info_token(name)
        if tok and tok in valid_tokens:
            inverted[tok].append(eid)

    s1_ids_list, ot_ids_list = [], []
    s1_valid = s1_df[s1_df["name_clean_legal"] != ""]
    for s1_id, name in zip(s1_valid["entity_id"], s1_valid["name_clean_legal"]):
        tok = first_info_token(name)
        if tok and tok in inverted:
            cands = inverted[tok][:max_cand_per_s1]
            for oid in cands:
                s1_ids_list.append(s1_id)
                ot_ids_list.append(oid)

    cand_df = pd.DataFrame({
        "s1_entity_id": s1_ids_list,
        "candidate_entity_id": ot_ids_list,
        "block": block_name,
    })
    print(f"[{block_name}] {len(cand_df):,} pairs ({time.time()-t0:.1f}s)")
    return cand_df


def block_last_informative_token(s1_df, other_df, max_df=300, max_cand_per_s1=50):
    """Block on the last informative token of name_clean_legal."""
    t0 = time.time()
    block_name = "last_info_token"

    STOP_TOKENS = {
        "the", "and", "for", "of", "in", "de", "del", "la", "el", "le", "les",
        "group", "services", "company", "international", "global", "solutions",
        "management", "systems", "technologies", "consulting", "enterprises",
        "partners", "associates", "holdings", "industries", "products", "national",
        "general", "american", "business", "financial", "capital", "investments",
    }

    def last_info_token(name):
        if not name:
            return ""
        tokens = name.split()
        for t in reversed(tokens):
            if len(t) >= 3 and t not in STOP_TOKENS:
                return t
        return ""

    token_counts = Counter()
    other_valid = other_df[other_df["name_clean_legal"] != ""]
    for name in other_valid["name_clean_legal"]:
        tok = last_info_token(name)
        if tok:
            token_counts[tok] += 1

    valid_tokens = {t for t, c in token_counts.items() if c <= max_df}

    inverted = defaultdict(list)
    for eid, name in zip(other_valid["entity_id"], other_valid["name_clean_legal"]):
        tok = last_info_token(name)
        if tok and tok in valid_tokens:
            inverted[tok].append(eid)

    s1_ids_list, ot_ids_list = [], []
    s1_valid = s1_df[s1_df["name_clean_legal"] != ""]
    for s1_id, name in zip(s1_valid["entity_id"], s1_valid["name_clean_legal"]):
        tok = last_info_token(name)
        if tok and tok in inverted:
            cands = inverted[tok][:max_cand_per_s1]
            for oid in cands:
                s1_ids_list.append(s1_id)
                ot_ids_list.append(oid)

    cand_df = pd.DataFrame({
        "s1_entity_id": s1_ids_list,
        "candidate_entity_id": ot_ids_list,
        "block": block_name,
    })
    print(f"[{block_name}] {len(cand_df):,} pairs ({time.time()-t0:.1f}s)")
    return cand_df


def block_token_pairs(s1_df, other_df, max_df=100, max_cand_per_s1=60):
    """Block on pairs of informative name tokens."""
    t0 = time.time()
    block_name = "token_pair"

    STOP_TOKENS = {
        "the", "and", "for", "of", "in", "de", "del", "la", "el", "le",
        "group", "services", "company", "international", "global", "solutions",
        "management", "systems", "technologies", "consulting",
    }

    def get_info_tokens(name):
        if not name:
            return []
        return [t for t in name.split() if len(t) >= 3 and t not in STOP_TOKENS]

    def get_token_pairs(name):
        toks = get_info_tokens(name)
        if len(toks) < 2:
            return []
        pairs = []
        toks_sorted = sorted(toks)  # Canonical order
        for i in range(len(toks_sorted)):
            for j in range(i+1, min(i+4, len(toks_sorted))):  # limit to nearby pairs
                pairs.append(f"{toks_sorted[i]}|{toks_sorted[j]}")
        return pairs

    # Count frequencies
    pair_counts = Counter()
    other_valid = other_df[other_df["name_clean_legal"] != ""]
    for name in other_valid["name_clean_legal"]:
        for pair in set(get_token_pairs(name)):
            pair_counts[pair] += 1

    valid_pairs = {p for p, c in pair_counts.items() if c <= max_df}

    # Build inverted index
    inverted = defaultdict(list)
    for eid, name in zip(other_valid["entity_id"], other_valid["name_clean_legal"]):
        for pair in set(get_token_pairs(name)):
            if pair in valid_pairs:
                inverted[pair].append(eid)

    s1_ids_list, ot_ids_list = [], []
    s1_valid = s1_df[s1_df["name_clean_legal"] != ""]
    for s1_id, name in zip(s1_valid["entity_id"], s1_valid["name_clean_legal"]):
        cand_set = set()
        for pair in set(get_token_pairs(name)):
            if pair in inverted:
                cand_set.update(inverted[pair])
                if len(cand_set) >= max_cand_per_s1:
                    break
        # Deterministic: sort by entity_id
        for oid in sorted(cand_set)[:max_cand_per_s1]:
            s1_ids_list.append(s1_id)
            ot_ids_list.append(oid)

    cand_df = pd.DataFrame({
        "s1_entity_id": s1_ids_list,
        "candidate_entity_id": ot_ids_list,
        "block": block_name,
    })
    print(f"[{block_name}] {len(cand_df):,} pairs ({time.time()-t0:.1f}s)")
    return cand_df


def block_char_prefix(s1_df, other_df, prefix_len=4, max_key_df=300, max_cand_per_s1=40):
    """Block on first N chars of name_clean_legal (shorter prefix than pass 8)."""
    t0 = time.time()
    block_name = f"char_prefix_{prefix_len}"

    s1_valid = s1_df[s1_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()
    other_valid = other_df[other_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()

    s1_valid["pkey"] = s1_valid["name_clean_legal"].str[:prefix_len]
    other_valid["pkey"] = other_valid["name_clean_legal"].str[:prefix_len]

    s1_valid = s1_valid[s1_valid["pkey"].str.len() >= prefix_len]
    other_valid = other_valid[other_valid["pkey"].str.len() >= prefix_len]

    key_counts = other_valid.groupby("pkey").size()
    valid_keys = key_counts[key_counts <= max_key_df].index
    other_valid = other_valid[other_valid["pkey"].isin(valid_keys)]
    s1_valid = s1_valid[s1_valid["pkey"].isin(valid_keys)]

    merged = s1_valid[["entity_id", "pkey"]].merge(
        other_valid[["entity_id", "pkey"]], on="pkey", suffixes=("_s1", "_other")
    )
    merged = merged.groupby("entity_id_s1").head(max_cand_per_s1)

    cand_df = merged[["entity_id_s1", "entity_id_other"]].rename(
        columns={"entity_id_s1": "s1_entity_id", "entity_id_other": "candidate_entity_id"}
    )
    cand_df["block"] = block_name
    print(f"[{block_name}] {len(cand_df):,} pairs ({time.time()-t0:.1f}s)")
    return cand_df


def block_name_token_no_country(s1_df, other_df, max_df=200, max_cand_per_s1=50, min_token_len=4):
    """Like country+name_token but WITHOUT country constraint — pure informative token matching."""
    t0 = time.time()
    block_name = "name_token_nocountry"

    STOP_TOKENS = {
        "the", "and", "for", "group", "services", "company", "international",
        "global", "solutions", "management", "systems", "technologies",
        "consulting", "enterprises", "partners", "associates", "holdings",
        "industries", "products", "national", "general", "american",
        "business", "financial", "capital", "investments", "properties",
        "construction", "development", "insurance", "marketing", "trading",
        "logistics", "communications", "engineering", "electric", "energy",
    }

    token_counts = Counter()
    other_valid = other_df[other_df["name_clean_legal"] != ""]
    for name in other_valid["name_clean_legal"]:
        for token in set(name.split()):
            if len(token) >= min_token_len and token not in STOP_TOKENS:
                token_counts[token] += 1

    valid_tokens = {t for t, c in token_counts.items() if c <= max_df}

    inverted = defaultdict(list)
    for eid, name in zip(other_valid["entity_id"], other_valid["name_clean_legal"]):
        for token in set(name.split()):
            if token in valid_tokens:
                inverted[token].append(eid)

    s1_ids_list, ot_ids_list = [], []
    s1_valid = s1_df[s1_df["name_clean_legal"] != ""]
    for s1_id, name in zip(s1_valid["entity_id"], s1_valid["name_clean_legal"]):
        cand_set = set()
        for token in set(name.split()):
            if token in inverted:
                cand_set.update(inverted[token])
                if len(cand_set) >= max_cand_per_s1 * 3:
                    break
        # Deterministic ranking: sort and cap
        for oid in sorted(cand_set)[:max_cand_per_s1]:
            s1_ids_list.append(s1_id)
            ot_ids_list.append(oid)

    cand_df = pd.DataFrame({
        "s1_entity_id": s1_ids_list,
        "candidate_entity_id": ot_ids_list,
        "block": block_name,
    })
    print(f"[{block_name}] {len(cand_df):,} pairs ({time.time()-t0:.1f}s)")
    return cand_df


def block_relaxed_rare_tokens(s1_df, other_df, max_df=1500, max_cand_per_s1=80):
    """Rare token blocking with much higher DF threshold and deterministic ranking."""
    t0 = time.time()
    block_name = "rare_token_relaxed"

    token_counts = Counter()
    other_valid = other_df[other_df["name_norm"] != ""]
    for name in other_valid["name_norm"]:
        for token in set(name.split()):
            if len(token) >= 3:
                token_counts[token] += 1

    valid_tokens = {t for t, c in token_counts.items() if 1 <= c <= max_df}

    inverted = defaultdict(list)
    for eid, name in zip(other_valid["entity_id"], other_valid["name_norm"]):
        for token in set(name.split()):
            if token in valid_tokens:
                inverted[token].append(eid)

    # For deterministic ranking, count shared token matches
    s1_ids_list, ot_ids_list = [], []
    s1_valid = s1_df[s1_df["name_norm"] != ""]
    for s1_id, name in zip(s1_valid["entity_id"], s1_valid["name_norm"]):
        cand_counts = Counter()
        for token in set(name.split()):
            if token in inverted:
                for oid in inverted[token]:
                    cand_counts[oid] += 1

        # Rank by number of shared tokens (descending), then by ID (stable)
        ranked = sorted(cand_counts.items(), key=lambda x: (-x[1], x[0]))
        for oid, _ in ranked[:max_cand_per_s1]:
            s1_ids_list.append(s1_id)
            ot_ids_list.append(oid)

    cand_df = pd.DataFrame({
        "s1_entity_id": s1_ids_list,
        "candidate_entity_id": ot_ids_list,
        "block": block_name,
    })
    print(f"[{block_name}] {len(cand_df):,} pairs ({time.time()-t0:.1f}s)")
    return cand_df


def block_tfidf_relaxed(s1_df, other_df, min_sim=0.35, top_k=20, sample_limit=300000):
    """TF-IDF char 3-gram with even more relaxed threshold."""
    t0 = time.time()
    block_name = f"char_ngram_{min_sim:.2f}"

    s1_valid = s1_df[s1_df["name_clean_legal"] != ""].copy()
    other_valid = other_df[other_df["name_clean_legal"] != ""].copy()

    if len(s1_valid) > sample_limit:
        s1_sub = s1_valid.head(sample_limit)
    else:
        s1_sub = s1_valid

    if len(other_valid) > sample_limit:
        other_sub = other_valid.head(sample_limit)
    else:
        other_sub = other_valid

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import awesome_cossim_topn

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2)
    corpus = pd.concat([s1_sub["name_clean_legal"], other_sub["name_clean_legal"]])
    vectorizer.fit(corpus)

    X_s1 = vectorizer.transform(s1_sub["name_clean_legal"])
    X_other = vectorizer.transform(other_sub["name_clean_legal"])

    top_sim = awesome_cossim_topn(X_s1, X_other.T, ntop=top_k, lower_bound=min_sim,
                                   use_threads=True, n_jobs=4)

    coo = top_sim.tocoo()
    s1_ids = s1_sub["entity_id"].values
    other_ids = other_sub["entity_id"].values

    cand_df = pd.DataFrame({
        "s1_entity_id": s1_ids[coo.row],
        "candidate_entity_id": other_ids[coo.col],
        "block": block_name,
    })

    del top_sim, coo, X_s1, X_other
    gc.collect()

    print(f"[{block_name}] {len(cand_df):,} pairs ({time.time()-t0:.1f}s)")
    return cand_df


def block_name_token_postal(s1_df, other_df, max_cand_per_s1=40):
    """Compound blocking: informative name token + postal/numeric address token."""
    t0 = time.time()
    block_name = "name_token_postal"

    STOP_TOKENS = {
        "the", "and", "for", "group", "services", "company", "international",
        "global", "solutions", "management", "systems", "technologies",
        "consulting", "enterprises", "partners", "associates", "holdings",
    }

    def get_compound_keys(name, addr):
        keys = []
        if not name or not addr:
            return keys
        name_tokens = [t for t in name.split() if len(t) >= 4 and t not in STOP_TOKENS]
        addr_nums = [t for t in addr.split() if any(c.isdigit() for c in t) and len(t) >= 3]
        for nt in name_tokens[:3]:
            for an in addr_nums[:3]:
                keys.append(f"{nt}|{an}")
        return keys

    # Count key frequencies
    key_counts = Counter()
    other_valid = other_df[(other_df["name_clean_legal"] != "") & (other_df["address_norm"] != "")]
    for name, addr in zip(other_valid["name_clean_legal"], other_valid["address_norm"]):
        for k in set(get_compound_keys(name, addr)):
            key_counts[k] += 1

    valid_keys = {k for k, c in key_counts.items() if c <= 200}

    inverted = defaultdict(list)
    for eid, name, addr in zip(other_valid["entity_id"], other_valid["name_clean_legal"], other_valid["address_norm"]):
        for k in set(get_compound_keys(name, addr)):
            if k in valid_keys:
                inverted[k].append(eid)

    s1_ids_list, ot_ids_list = [], []
    s1_valid = s1_df[(s1_df["name_clean_legal"] != "") & (s1_df["address_norm"] != "")]
    for s1_id, name, addr in zip(s1_valid["entity_id"], s1_valid["name_clean_legal"], s1_valid["address_norm"]):
        cand_set = set()
        for k in set(get_compound_keys(name, addr)):
            if k in inverted:
                cand_set.update(inverted[k])
        for oid in sorted(cand_set)[:max_cand_per_s1]:
            s1_ids_list.append(s1_id)
            ot_ids_list.append(oid)

    cand_df = pd.DataFrame({
        "s1_entity_id": s1_ids_list,
        "candidate_entity_id": ot_ids_list,
        "block": block_name,
    })
    print(f"[{block_name}] {len(cand_df):,} pairs ({time.time()-t0:.1f}s)")
    return cand_df


def block_phonetic_extended(s1_df, other_df, max_key_df=500, max_cand_per_s1=60):
    """Extended phonetic blocking: uses first 3 tokens and more lenient DF."""
    t0 = time.time()
    block_name = "phonetic_ext"

    s1_valid = s1_df[s1_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()
    other_valid = other_df[other_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()

    def make_phonetic_key(name):
        tokens = name.split()
        codes = []
        for t in tokens[:3]:
            code = _double_metaphone_simple(t)
            if code:
                codes.append(code)
        return "|".join(codes) if codes else ""

    s1_valid["phon_key"] = s1_valid["name_clean_legal"].apply(make_phonetic_key)
    other_valid["phon_key"] = other_valid["name_clean_legal"].apply(make_phonetic_key)

    s1_valid = s1_valid[s1_valid["phon_key"] != ""]
    other_valid = other_valid[other_valid["phon_key"] != ""]

    key_counts = other_valid.groupby("phon_key").size()
    valid_keys = key_counts[key_counts <= max_key_df].index
    other_valid = other_valid[other_valid["phon_key"].isin(valid_keys)]
    s1_valid = s1_valid[s1_valid["phon_key"].isin(valid_keys)]

    merged = s1_valid[["entity_id", "phon_key"]].merge(
        other_valid[["entity_id", "phon_key"]], on="phon_key", suffixes=("_s1", "_other")
    )
    merged = merged.groupby("entity_id_s1").head(max_cand_per_s1)

    cand_df = merged[["entity_id_s1", "entity_id_other"]].rename(
        columns={"entity_id_s1": "s1_entity_id", "entity_id_other": "candidate_entity_id"}
    )
    cand_df["block"] = block_name
    print(f"[{block_name}] {len(cand_df):,} pairs ({time.time()-t0:.1f}s)")
    return cand_df


def block_name_prefix_4(s1_df, other_df, max_key_df=500, max_cand_per_s1=40):
    """Shorter 4-char prefix blocking."""
    return block_char_prefix(s1_df, other_df, prefix_len=4, max_key_df=max_key_df, max_cand_per_s1=max_cand_per_s1)


def block_sorted_tokens_relaxed(s1_df, other_df, max_key_df=800, max_cand_per_s1=60):
    """Sorted token blocking with relaxed parameters."""
    t0 = time.time()
    block_name = "sorted_token_relaxed"

    s1_valid = s1_df[s1_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()
    other_valid = other_df[other_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()

    s1_valid["sorted_key"] = s1_valid["name_clean_legal"].str.split().apply(
        lambda tokens: " ".join(sorted(tokens)) if isinstance(tokens, list) and len(tokens) >= 2 else ""
    )
    other_valid["sorted_key"] = other_valid["name_clean_legal"].str.split().apply(
        lambda tokens: " ".join(sorted(tokens)) if isinstance(tokens, list) and len(tokens) >= 2 else ""
    )

    s1_valid = s1_valid[s1_valid["sorted_key"].str.len() >= 4]
    other_valid = other_valid[other_valid["sorted_key"].str.len() >= 4]

    key_counts = other_valid.groupby("sorted_key").size()
    valid_keys = key_counts[key_counts <= max_key_df].index
    other_valid = other_valid[other_valid["sorted_key"].isin(valid_keys)]
    s1_valid = s1_valid[s1_valid["sorted_key"].isin(valid_keys)]

    merged = s1_valid[["entity_id", "sorted_key"]].merge(
        other_valid[["entity_id", "sorted_key"]], on="sorted_key", suffixes=("_s1", "_other")
    )
    merged = merged.groupby("entity_id_s1").head(max_cand_per_s1)

    cand_df = merged[["entity_id_s1", "entity_id_other"]].rename(
        columns={"entity_id_s1": "s1_entity_id", "entity_id_other": "candidate_entity_id"}
    )
    cand_df["block"] = block_name
    print(f"[{block_name}] {len(cand_df):,} pairs ({time.time()-t0:.1f}s)")
    return cand_df


def block_consonant_skeleton(s1_df, other_df, max_key_df=300, max_cand_per_s1=50):
    """Block on consonant skeleton of name (strip vowels after first char)."""
    t0 = time.time()
    block_name = "consonant_skeleton"

    VOWELS = set("aeiou")

    def consonant_key(name):
        if not name or len(name) < 3:
            return ""
        # Remove spaces, keep first char, strip vowels from rest
        stripped = name.replace(" ", "")
        result = stripped[0] + "".join(c for c in stripped[1:] if c not in VOWELS)
        return result[:10] if len(result) >= 4 else ""

    s1_valid = s1_df[s1_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()
    other_valid = other_df[other_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()

    s1_valid["cons_key"] = s1_valid["name_clean_legal"].apply(consonant_key)
    other_valid["cons_key"] = other_valid["name_clean_legal"].apply(consonant_key)

    s1_valid = s1_valid[s1_valid["cons_key"] != ""]
    other_valid = other_valid[other_valid["cons_key"] != ""]

    key_counts = other_valid.groupby("cons_key").size()
    valid_keys = key_counts[key_counts <= max_key_df].index
    other_valid = other_valid[other_valid["cons_key"].isin(valid_keys)]
    s1_valid = s1_valid[s1_valid["cons_key"].isin(valid_keys)]

    merged = s1_valid[["entity_id", "cons_key"]].merge(
        other_valid[["entity_id", "cons_key"]], on="cons_key", suffixes=("_s1", "_other")
    )
    merged = merged.groupby("entity_id_s1").head(max_cand_per_s1)

    cand_df = merged[["entity_id_s1", "entity_id_other"]].rename(
        columns={"entity_id_s1": "s1_entity_id", "entity_id_other": "candidate_entity_id"}
    )
    cand_df["block"] = block_name
    print(f"[{block_name}] {len(cand_df):,} pairs ({time.time()-t0:.1f}s)")
    return cand_df


def block_name_word_tfidf(s1_df, other_df, min_sim=0.50, top_k=15, sample_limit=300000):
    """TF-IDF word-level (not char-ngram) blocking for different surface coverage."""
    t0 = time.time()
    block_name = f"word_tfidf_{min_sim:.2f}"

    s1_valid = s1_df[s1_df["name_clean_legal"] != ""].copy()
    other_valid = other_df[other_df["name_clean_legal"] != ""].copy()

    if len(s1_valid) > sample_limit:
        s1_sub = s1_valid.head(sample_limit)
    else:
        s1_sub = s1_valid

    if len(other_valid) > sample_limit:
        other_sub = other_valid.head(sample_limit)
    else:
        other_sub = other_valid

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import awesome_cossim_topn

    vectorizer = TfidfVectorizer(analyzer="word", min_df=2, max_df=0.1)
    corpus = pd.concat([s1_sub["name_clean_legal"], other_sub["name_clean_legal"]])
    vectorizer.fit(corpus)

    X_s1 = vectorizer.transform(s1_sub["name_clean_legal"])
    X_other = vectorizer.transform(other_sub["name_clean_legal"])

    top_sim = awesome_cossim_topn(X_s1, X_other.T, ntop=top_k, lower_bound=min_sim,
                                   use_threads=True, n_jobs=4)

    coo = top_sim.tocoo()
    s1_ids = s1_sub["entity_id"].values
    other_ids = other_sub["entity_id"].values

    cand_df = pd.DataFrame({
        "s1_entity_id": s1_ids[coo.row],
        "candidate_entity_id": other_ids[coo.col],
        "block": block_name,
    })

    del top_sim, coo, X_s1, X_other
    gc.collect()

    print(f"[{block_name}] {len(cand_df):,} pairs ({time.time()-t0:.1f}s)")
    return cand_df


# ================================================================
# PHASE 5/6: EVALUATION AND COMBINATION
# ================================================================

def evaluate_combination(block_dfs, gt_dict, total_gt, val_s1, label=""):
    """Evaluate a combination of blocking passes."""
    # Union all candidates
    all_pairs = set()
    for bdf in block_dfs:
        if bdf.empty:
            continue
        for s1_id, ot_id in zip(bdf["s1_entity_id"], bdf["candidate_entity_id"]):
            all_pairs.add((s1_id, ot_id))

    # Count GT found
    gt_found = set()
    for s1_id, ot_id in all_pairs:
        if s1_id in gt_dict and ot_id in gt_dict[s1_id]:
            gt_found.add((s1_id, ot_id))

    total_cands = len(all_pairs)
    recall = len(gt_found) / total_gt if total_gt > 0 else 0.0
    avg_per_s1 = total_cands / len(val_s1) if len(val_s1) > 0 else 0.0

    # Entity-level coverage
    gt_per_s1 = defaultdict(int)
    found_per_s1 = defaultdict(int)
    for s1_id in gt_dict:
        gt_per_s1[s1_id] = len(gt_dict[s1_id])
    for s1_id, _ in gt_found:
        found_per_s1[s1_id] += 1

    complete = sum(1 for s1_id in gt_per_s1 if found_per_s1.get(s1_id, 0) == gt_per_s1[s1_id])
    zero = sum(1 for s1_id in gt_per_s1 if found_per_s1.get(s1_id, 0) == 0)
    total_gt_s1 = len(gt_per_s1)

    return {
        "label": label,
        "total_candidates": total_cands,
        "gt_found": len(gt_found),
        "total_gt": total_gt,
        "recall": recall,
        "avg_per_s1": avg_per_s1,
        "complete_coverage_pct": complete / total_gt_s1 * 100 if total_gt_s1 > 0 else 0.0,
        "zero_coverage_pct": zero / total_gt_s1 * 100 if total_gt_s1 > 0 else 0.0,
        "found_pairs": gt_found,
    }


# ================================================================
# MAIN PIPELINE
# ================================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("V3 BLOCKING BENCHMARK — COMPREHENSIVE ANALYSIS & IMPROVEMENT")
    print("=" * 70)

    # ── Phase 1: Validation Benchmark ──
    val_s1, s2, s3, val_gt_s2, val_gt_s3, val_total_s2, val_total_s3 = load_validation_benchmark()

    print(f"\nValidation benchmark loaded:")
    print(f"  S1 sample: {len(val_s1):,}")
    print(f"  S2 total:  {len(s2):,}")
    print(f"  S3 total:  {len(s3):,}")
    print(f"  GT S2 pairs: {val_total_s2:,}")
    print(f"  GT S3 pairs: {val_total_s3:,}")

    all_results = {}

    for source_label, other_df, gt_dict, total_gt in [
        ("S2", s2, val_gt_s2, val_total_s2),
        ("S3", s3, val_gt_s3, val_total_s3),
    ]:
        print(f"\n\n{'#' * 70}")
        print(f"# PROCESSING S1 → {source_label}")
        print(f"{'#' * 70}")

        # ── Phase 2: V1 passes with per-pass metrics ──
        print(f"\n{'=' * 70}")
        print(f"PHASE 2: V1 BLOCKING PASSES — {source_label}")
        print(f"{'=' * 70}")

        v1_blocks, v1_names = run_v1_passes(val_s1, other_df, source_label)

        v1_incremental, v1_found = evaluate_incremental(
            v1_blocks, v1_names, gt_dict, total_gt
        )

        print(f"\n{'Pass':<30} {'Cands':>10} {'GT Found':>10} {'New GT':>8} {'Cum GT':>8} {'Inc Rec%':>8} {'Cum Rec%':>8}")
        print("-" * 90)
        for r in v1_incremental:
            print(f"{r['pass']:<30} {r['candidates']:>10,} {r['gt_found']:>10,} {r['new_gt']:>8,} "
                  f"{r['cumulative_gt']:>8,} {r['incremental_recall']*100:>7.2f}% {r['cumulative_recall']*100:>7.2f}%")

        v1_eval = evaluate_combination(v1_blocks, gt_dict, total_gt, val_s1, f"V1_{source_label}")
        print(f"\nV1 Union — {source_label}:")
        print(f"  Recall:     {v1_eval['recall']*100:.2f}% ({v1_eval['gt_found']:,}/{total_gt:,})")
        print(f"  Candidates: {v1_eval['total_candidates']:,}")
        print(f"  Avg/S1:     {v1_eval['avg_per_s1']:.1f}")
        print(f"  Complete:   {v1_eval['complete_coverage_pct']:.2f}%")
        print(f"  Zero:       {v1_eval['zero_coverage_pct']:.2f}%")

        # ── Phase 3: False Negative Analysis ──
        fn_categories = analyze_false_negatives(
            val_s1, other_df, gt_dict, total_gt,
            v1_found, source_label
        )

        # ── Phase 4: New Blocking Strategies ──
        print(f"\n{'=' * 70}")
        print(f"PHASE 4: NEW BLOCKING STRATEGIES — {source_label}")
        print(f"{'=' * 70}")

        new_blocks = {}

        # 1. First informative token
        new_blocks["first_info_token"] = block_first_informative_token(
            val_s1, other_df, max_df=300, max_cand_per_s1=50)

        # 2. Last informative token
        new_blocks["last_info_token"] = block_last_informative_token(
            val_s1, other_df, max_df=300, max_cand_per_s1=50)

        # 3. Token pairs
        new_blocks["token_pair"] = block_token_pairs(
            val_s1, other_df, max_df=100, max_cand_per_s1=60)

        # 4. 4-char prefix
        new_blocks["char_prefix_4"] = block_name_prefix_4(
            val_s1, other_df, max_key_df=500, max_cand_per_s1=40)

        # 5. Name token without country
        new_blocks["name_token_nocountry"] = block_name_token_no_country(
            val_s1, other_df, max_df=200, max_cand_per_s1=50, min_token_len=4)

        # 6. Relaxed rare tokens (higher DF)
        new_blocks["rare_token_relaxed"] = block_relaxed_rare_tokens(
            val_s1, other_df, max_df=1500, max_cand_per_s1=80)

        # 7. TF-IDF even more relaxed
        new_blocks["tfidf_0.35"] = block_tfidf_relaxed(
            val_s1, other_df, min_sim=0.35, top_k=20, sample_limit=300000)

        # 8. Name token + postal
        new_blocks["name_token_postal"] = block_name_token_postal(
            val_s1, other_df, max_cand_per_s1=40)

        # 9. Extended phonetic
        new_blocks["phonetic_ext"] = block_phonetic_extended(
            val_s1, other_df, max_key_df=500, max_cand_per_s1=60)

        # 10. Consonant skeleton
        new_blocks["consonant_skeleton"] = block_consonant_skeleton(
            val_s1, other_df, max_key_df=300, max_cand_per_s1=50)

        # 11. Sorted tokens relaxed
        new_blocks["sorted_token_relaxed"] = block_sorted_tokens_relaxed(
            val_s1, other_df, max_key_df=800, max_cand_per_s1=60)

        # 12. Word-level TF-IDF
        new_blocks["word_tfidf_0.50"] = block_name_word_tfidf(
            val_s1, other_df, min_sim=0.50, top_k=15)

        # Evaluate each new blocker individually AND incrementally on top of V1
        print(f"\n{'=' * 70}")
        print(f"PHASE 6: INDIVIDUAL NEW BLOCKER EVALUATION — {source_label}")
        print(f"{'=' * 70}")

        print(f"\n{'Blocker':<25} {'Cands':>10} {'GT Found':>10} {'NEW vs V1':>10} {'Inc Rec%':>8}")
        print("-" * 70)

        new_blocker_stats = []
        for bname, bdf in new_blocks.items():
            eval_r = evaluate_single_block(bdf, gt_dict, total_gt, bname)

            # New GT pairs not in V1
            pass_found = set()
            if not bdf.empty:
                for s1_id, ot_id in zip(bdf["s1_entity_id"], bdf["candidate_entity_id"]):
                    if s1_id in gt_dict and ot_id in gt_dict[s1_id]:
                        pass_found.add((s1_id, ot_id))
            new_vs_v1 = pass_found - v1_found

            print(f"{bname:<25} {eval_r['candidates']:>10,} {eval_r['gt_found']:>10,} "
                  f"{len(new_vs_v1):>10,} {len(new_vs_v1)/total_gt*100:>7.2f}%")

            new_blocker_stats.append({
                "name": bname,
                "candidates": eval_r["candidates"],
                "gt_found": eval_r["gt_found"],
                "new_vs_v1": len(new_vs_v1),
                "incremental_recall": len(new_vs_v1) / total_gt if total_gt > 0 else 0.0,
                "new_pairs": new_vs_v1,
            })

        # ── Phase 7: Ablation / Combination Search ──
        print(f"\n{'=' * 70}")
        print(f"PHASE 7: COMBINATION SEARCH — {source_label}")
        print(f"{'=' * 70}")

        # Sort new blockers by incremental contribution
        new_blocker_stats.sort(key=lambda x: -x["new_vs_v1"])

        # Build V3 combination: V1 + best new blockers
        # Start with V1 baseline, add new blockers greedily by incremental GT recovery
        v3_blocks = list(v1_blocks)  # Start with all V1
        v3_found = set(v1_found)
        v3_selected = []

        for nbs in new_blocker_stats:
            new_pairs = nbs["new_pairs"] - v3_found
            if len(new_pairs) >= 5:  # Only add if it recovers at least 5 new GT pairs
                v3_blocks.append(new_blocks[nbs["name"]])
                v3_found.update(nbs["new_pairs"])
                v3_selected.append({
                    "name": nbs["name"],
                    "new_gt_added": len(new_pairs),
                    "candidates": nbs["candidates"],
                })
                print(f"  + Added {nbs['name']}: +{len(new_pairs)} new GT pairs")

        v3_eval = evaluate_combination(v3_blocks, gt_dict, total_gt, val_s1, f"V3_{source_label}")

        print(f"\n{'=' * 70}")
        print(f"V3 COMBINATION RESULTS — {source_label}")
        print(f"{'=' * 70}")
        print(f"  V1 Recall:     {v1_eval['recall']*100:.2f}%")
        print(f"  V3 Recall:     {v3_eval['recall']*100:.2f}% (+{(v3_eval['recall']-v1_eval['recall'])*100:.2f}pp)")
        print(f"  V1 Candidates: {v1_eval['total_candidates']:,}")
        print(f"  V3 Candidates: {v3_eval['total_candidates']:,}")
        print(f"  V3 Avg/S1:     {v3_eval['avg_per_s1']:.1f}")
        print(f"  V3 Complete:   {v3_eval['complete_coverage_pct']:.2f}%")
        print(f"  V3 Zero:       {v3_eval['zero_coverage_pct']:.2f}%")
        print(f"  New blockers added: {len(v3_selected)}")
        for sel in v3_selected:
            print(f"    - {sel['name']}: +{sel['new_gt_added']} GT, {sel['candidates']:,} cands")

        all_results[source_label] = {
            "v1_eval": {k: v for k, v in v1_eval.items() if k != "found_pairs"},
            "v3_eval": {k: v for k, v in v3_eval.items() if k != "found_pairs"},
            "v1_incremental": v1_incremental,
            "new_blocker_stats": [{k: v for k, v in s.items() if k != "new_pairs"} for s in new_blocker_stats],
            "v3_selected": v3_selected,
            "fn_categories": {cat: len(items) for cat, items in fn_categories.items()} if fn_categories else {},
        }

        # Save V3 candidates
        all_cand_pairs = set()
        for bdf in v3_blocks:
            if not bdf.empty:
                for s1_id, ot_id in zip(bdf["s1_entity_id"], bdf["candidate_entity_id"]):
                    all_cand_pairs.add((s1_id, ot_id))

        v3_cands_df = pd.DataFrame(list(all_cand_pairs), columns=["s1_idx", f"{source_label.lower()}_idx"])
        v3_cands_df.sort_values(["s1_idx", f"{source_label.lower()}_idx"], inplace=True)
        v3_cands_df.reset_index(drop=True, inplace=True)
        v3_cands_df.to_parquet(V3_DIR / f"val_v3_s1_{source_label.lower()}_candidates.parquet", index=False)
        print(f"\nSaved V3 candidates: {V3_DIR / f'val_v3_s1_{source_label.lower()}_candidates.parquet'}")

        del v1_blocks, v3_blocks, new_blocks
        gc.collect()

    # ── Phase 10: Final Report ──
    print(f"\n\n{'=' * 70}")
    print("PHASE 10: FINAL REPORT")
    print("=" * 70)

    for src in ["S2", "S3"]:
        r = all_results[src]
        print(f"\n--- {src} ---")
        print(f"  V1 Recall:       {r['v1_eval']['recall']*100:.2f}%  ({r['v1_eval']['gt_found']:,}/{r['v1_eval']['total_gt']:,})")
        print(f"  V3 Recall:       {r['v3_eval']['recall']*100:.2f}%  ({r['v3_eval']['gt_found']:,}/{r['v3_eval']['total_gt']:,})")
        print(f"  V1 Candidates:   {r['v1_eval']['total_candidates']:,}  (avg {r['v1_eval']['avg_per_s1']:.1f}/S1)")
        print(f"  V3 Candidates:   {r['v3_eval']['total_candidates']:,}  (avg {r['v3_eval']['avg_per_s1']:.1f}/S1)")
        print(f"  V3 Complete:     {r['v3_eval']['complete_coverage_pct']:.2f}%")
        print(f"  V3 Zero:         {r['v3_eval']['zero_coverage_pct']:.2f}%")
        print(f"  Improvement:     +{(r['v3_eval']['recall']-r['v1_eval']['recall'])*100:.2f}pp")
        if r["v3_selected"]:
            print(f"  New blockers ({len(r['v3_selected'])}):")
            for sel in r["v3_selected"]:
                print(f"    - {sel['name']}: +{sel['new_gt_added']} GT, {sel['candidates']:,} cands")

    # Save comprehensive results
    save_results = {}
    for src in ["S2", "S3"]:
        r = all_results[src]
        save_results[src] = {
            "v1": r["v1_eval"],
            "v3": r["v3_eval"],
            "v1_per_pass": r["v1_incremental"],
            "new_blockers": r["new_blocker_stats"],
            "v3_selected": r["v3_selected"],
            "false_negative_categories": r["fn_categories"],
        }

    with open(V3_DIR / "v3_blocking_benchmark_summary.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"V3 BENCHMARK COMPLETE — Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"All outputs saved to: {V3_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
