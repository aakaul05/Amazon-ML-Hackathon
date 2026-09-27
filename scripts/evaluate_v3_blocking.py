"""
scripts/evaluate_v3_blocking.py
================================
Phase 3 & Phase 4: Implementation and Ablation Testing of Improved Blocking V3.

Evaluates on the deterministic 10,000 S1 validation benchmark:
  - val_s1_10k.parquet
  - val_gt_10k.json

Target:
  - Candidate Recall > 90% (Stretch: > 95%)
  - Controlled candidate volume (< 80 candidates / S1)
  - Zero arbitrary set truncation (deterministic tie-breaking)

New V3 Blockers addressing Phase 2 failure modes:
  1. address_num_distinctive_token: (house/building number + distinctive street token)
     -> Solves 52% of false negatives (disjoint names with strong address)
  2. compact_name_prefix: (space-stripped name prefix of 8 chars)
     -> Solves domain/concatenated name variations (e.g. 'smartservices... com' vs 'smart services...')
  3. rare_token_pairs: (two-token compound keys)
     -> Solves token subset/expansion and common word overlap
  4. name_prefix_plus_address_num: (4-char name prefix + house number)
     -> Solves name typos with exact address
  5. vowel_stripped_skeleton: (consonant skeleton prefix for spelling variations)
  6. improved_first_last_tokens: (frequency-guarded brand & sector tokens)
  7. partitioned_tfidf_relaxed: (character 3-gram partitioned by first character, min_sim=0.45)
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
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

OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "validation"
NORMALIZED_DIR = REPO_ROOT / "data" / "outputs" / "normalized_cache"

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
    "north", "south", "east", "west", "one", "two", "three", "alpha", "ltd", "inc", "corp", "llc", "pvt"
}

ADDRESS_STOPWORDS = {
    "road", "street", "st", "rd", "floor", "fl", "suite", "ste", "avenue", "ave",
    "building", "bldg", "lane", "ln", "nagar", "block", "blk", "sector", "sec",
    "house", "plot", "near", "opp", "post", "dist", "state", "city", "west",
    "east", "north", "south", "delhi", "mumbai", "india", "usa", "france",
    "colony", "marg", "vihar", "apartment", "apt", "phase", "cross", "main",
    "circle", "park", "bazaar", "hall", "point", "tower", "plaza", "complex"
}

# Bitmask values for V3 passes
V3_BLOCK_BITMASK = {
    "exact_name_norm":                 1,
    "exact_clean_legal":               2,
    "rare_token_ranked":               4,
    "address_num_distinctive_token":    8,
    "compact_name_prefix":             16,
    "rare_token_pairs":                32,
    "name_prefix_plus_addr_num":       64,
    "first_token":                    128,
    "last_token":                     256,
    "sorted_token":                   512,
    "phonetic":                      1024,
    "country_name_token":            2048,
    "partitioned_tfidf":             4096,
}


def build_s1_v3_profiles(val_s1: pd.DataFrame) -> dict:
    """Extracts all V3 indexing keys for S1 entities."""
    profiles = {}
    for _, row in val_s1.iterrows():
        eid = row["entity_id"]
        nn = str(row["name_norm"] or "")
        ncl = str(row["name_clean_legal"] or "")
        an = str(row["address_norm"] or "")
        cn = str(row["country_norm"] or "")

        # 1. Name tokens
        tokens_nn = [t for t in nn.split() if len(t) >= 3]
        tokens_ncl = [t for t in ncl.split() if len(t) >= 3 and t not in NAME_STOPWORDS]

        # 2. Compact name prefix (no spaces)
        compact = ncl.replace(" ", "")
        compact_prefix = compact[:8] if len(compact) >= 6 else ""

        # 3. First and last token
        first_tok = tokens_ncl[0] if len(tokens_ncl) >= 1 and len(tokens_ncl[0]) >= 4 else ""
        last_tok = tokens_ncl[-1] if len(tokens_ncl) >= 2 and len(tokens_ncl[-1]) >= 4 and tokens_ncl[-1] != first_tok else ""

        # 4. Token pairs (combinations of 2 informative tokens)
        token_pairs = []
        if len(tokens_ncl) >= 2:
            sorted_t = sorted(tokens_ncl[:4])
            for i in range(len(sorted_t)):
                for j in range(i + 1, len(sorted_t)):
                    token_pairs.append(f"{sorted_t[i]}_{sorted_t[j]}")

        # 5. Sorted key & 6-char prefix
        sorted_tokens = sorted(ncl.split())
        sorted_key = " ".join(sorted_tokens) if len(sorted_tokens) >= 2 and len(ncl) >= 4 else ""
        prefix_key = ncl[:6] if len(ncl) >= 6 else ""

        # 6. Address parsing: numbers & distinctive street words
        addr_tokens = an.split()
        nums = [t for t in addr_tokens if any(c.isdigit() for c in t) and len(t) >= 2]
        distinctive_addr = [t for t in addr_tokens if len(t) >= 4 and t not in ADDRESS_STOPWORDS and t.isalpha()]

        # Compound address keys: (num, distinctive_addr_word)
        addr_num_words = []
        if nums and distinctive_addr:
            for n in nums[:2]:
                for w in distinctive_addr[:3]:
                    addr_num_words.append(f"{n}|{w}")

        # 7. Name prefix (4 chars) + address number
        name_pfx_addr_num = []
        if len(ncl) >= 4 and nums:
            pfx4 = ncl[:4]
            for n in nums[:2]:
                name_pfx_addr_num.append(f"{pfx4}|{n}")

        # 8. Phonetic key
        phon_codes = [_double_metaphone_simple(t) for t in ncl.split()[:2] if _double_metaphone_simple(t)]
        phon_key = "|".join(phon_codes) if phon_codes else ""

        # 9. Country + token
        c_tokens = [(cn, t) for t in tokens_ncl if len(t) >= 4]

        profiles[eid] = {
            "entity_id": eid,
            "name_norm": nn,
            "name_clean_legal": ncl,
            "address_norm": an,
            "country_norm": cn,
            "tokens_nn": set(tokens_nn),
            "tokens_ncl": set(tokens_ncl),
            "compact_prefix": compact_prefix,
            "first_tok": first_tok,
            "last_tok": last_tok,
            "token_pairs": set(token_pairs),
            "sorted_key": sorted_key,
            "prefix_key": prefix_key,
            "addr_num_words": set(addr_num_words),
            "name_pfx_addr_num": set(name_pfx_addr_num),
            "phon_key": phon_key,
            "country_tokens": c_tokens,
        }
    return profiles


def run_v3_streaming_blocking(
    s1_df: pd.DataFrame,
    source_parquet_path: Path,
    source_name: str,
) -> Dict[str, Set[Tuple[str, str]]]:
    """Runs V3 streaming blocking on source parquet."""
    t0 = time.time()
    print(f"\n[{source_name}] Building S1 V3 profiles ({len(s1_df):,} entities)...")
    s1_profiles = build_s1_v3_profiles(s1_df)

    # Collect target keys needed by S1
    target_name_norm = {p["name_norm"] for p in s1_profiles.values() if p["name_norm"]}
    target_name_clean = {p["name_clean_legal"] for p in s1_profiles.values() if p["name_clean_legal"]}
    target_rare_tokens = {t for p in s1_profiles.values() for t in p["tokens_nn"]}
    target_compact = {p["compact_prefix"] for p in s1_profiles.values() if p["compact_prefix"]}
    target_token_pairs = {tp for p in s1_profiles.values() for tp in p["token_pairs"]}
    target_addr_nw = {k for p in s1_profiles.values() for k in p["addr_num_words"]}
    target_pfx_num = {k for p in s1_profiles.values() for k in p["name_pfx_addr_num"]}
    target_first_toks = {p["first_tok"] for p in s1_profiles.values() if p["first_tok"]}
    target_last_toks = {p["last_tok"] for p in s1_profiles.values() if p["last_tok"]}
    target_sorted_keys = {p["sorted_key"] for p in s1_profiles.values() if p["sorted_key"]}
    target_phon_keys = {p["phon_key"] for p in s1_profiles.values() if p["phon_key"]}
    target_country_toks = {ct for p in s1_profiles.values() for ct in p["country_tokens"]}

    # Inverted indices
    idx_name_norm = defaultdict(list)
    cnt_name_norm = Counter()

    idx_name_clean = defaultdict(list)
    cnt_name_clean = Counter()

    idx_rare_tokens = defaultdict(list)
    cnt_rare_tokens = Counter()

    idx_compact = defaultdict(list)
    cnt_compact = Counter()

    idx_token_pairs = defaultdict(list)
    cnt_token_pairs = Counter()

    idx_addr_nw = defaultdict(list)
    cnt_addr_nw = Counter()

    idx_pfx_num = defaultdict(list)
    cnt_pfx_num = Counter()

    idx_first_tok = defaultdict(list)
    cnt_first_tok = Counter()

    idx_last_tok = defaultdict(list)
    cnt_last_tok = Counter()

    idx_sorted_key = defaultdict(list)
    cnt_sorted_key = Counter()

    idx_phon_key = defaultdict(list)
    cnt_phon_key = Counter()

    idx_country_tok = defaultdict(list)
    cnt_country_tok = Counter()

    pfile = pq.ParquetFile(source_parquet_path)
    total_source_rows = pfile.metadata.num_rows
    print(f"[{source_name}] Streaming index scan across {total_source_rows:,} rows...")

    cols_to_read = ["entity_id", "name_norm", "name_clean_legal", "address_norm", "country_norm"]
    
    for batch in pfile.iter_batches(batch_size=500_000, columns=cols_to_read):
        df_b = batch.to_pandas()
        for ot_id, nn, ncl, an, cn in zip(
            df_b["entity_id"], df_b["name_norm"], df_b["name_clean_legal"], df_b["address_norm"], df_b["country_norm"]
        ):
            nn = str(nn or "")
            ncl = str(ncl or "")
            an = str(an or "")
            cn = str(cn or "")

            # 1. Exact name norm
            if nn in target_name_norm:
                cnt_name_norm[nn] += 1
                if len(idx_name_norm[nn]) < 80:
                    idx_name_norm[nn].append(ot_id)

            # 2. Exact clean legal
            if ncl in target_name_clean:
                cnt_name_clean[ncl] += 1
                if len(idx_name_clean[ncl]) < 80:
                    idx_name_clean[ncl].append(ot_id)

            # 3. Rare tokens
            words_nn = set(nn.split())
            for t in words_nn:
                if t in target_rare_tokens:
                    cnt_rare_tokens[t] += 1
                    if len(idx_rare_tokens[t]) < 100:
                        idx_rare_tokens[t].append(ot_id)

            words_ncl = ncl.split()
            # 4. Compact name prefix (no spaces)
            compact = ncl.replace(" ", "")
            if len(compact) >= 6:
                cp = compact[:8]
                if cp in target_compact:
                    cnt_compact[cp] += 1
                    if len(idx_compact[cp]) < 60:
                        idx_compact[cp].append(ot_id)

            # 5. Token pairs
            if len(words_ncl) >= 2:
                filtered_t = [w for w in words_ncl[:4] if len(w) >= 3 and w not in NAME_STOPWORDS]
                sorted_t = sorted(filtered_t)
                for i in range(len(sorted_t)):
                    for j in range(i + 1, len(sorted_t)):
                        tp = f"{sorted_t[i]}_{sorted_t[j]}"
                        if tp in target_token_pairs:
                            cnt_token_pairs[tp] += 1
                            if len(idx_token_pairs[tp]) < 60:
                                idx_token_pairs[tp].append(ot_id)

            # 6. Address: numbers + distinctive words
            if an:
                a_tokens = an.split()
                nums = [t for t in a_tokens if any(c.isdigit() for c in t) and len(t) >= 2]
                dist_addr = [t for t in a_tokens if len(t) >= 4 and t not in ADDRESS_STOPWORDS and t.isalpha()]
                if nums and dist_addr:
                    for n in nums[:2]:
                        for w in dist_addr[:3]:
                            nw_key = f"{n}|{w}"
                            if nw_key in target_addr_nw:
                                cnt_addr_nw[nw_key] += 1
                                if len(idx_addr_nw[nw_key]) < 60:
                                    idx_addr_nw[nw_key].append(ot_id)

                # 7. Name prefix (4) + address num
                if len(ncl) >= 4 and nums:
                    pfx4 = ncl[:4]
                    for n in nums[:2]:
                        pn_key = f"{pfx4}|{n}"
                        if pn_key in target_pfx_num:
                            cnt_pfx_num[pn_key] += 1
                            if len(idx_pfx_num[pn_key]) < 60:
                                idx_pfx_num[pn_key].append(ot_id)

            # 8. First & Last token
            if words_ncl:
                if words_ncl[0] in target_first_toks:
                    cnt_first_tok[words_ncl[0]] += 1
                    if len(idx_first_tok[words_ncl[0]]) < 80:
                        idx_first_tok[words_ncl[0]].append(ot_id)
                if len(words_ncl) >= 2 and words_ncl[-1] in target_last_toks:
                    cnt_last_tok[words_ncl[-1]] += 1
                    if len(idx_last_tok[words_ncl[-1]]) < 80:
                        idx_last_tok[words_ncl[-1]].append(ot_id)

            # 9. Sorted token
            if len(words_ncl) >= 2:
                s_key = " ".join(sorted(words_ncl))
                if s_key in target_sorted_keys:
                    cnt_sorted_key[s_key] += 1
                    if len(idx_sorted_key[s_key]) < 80:
                        idx_sorted_key[s_key].append(ot_id)

            # 10. Phonetic
            if words_ncl:
                codes = [_double_metaphone_simple(t) for t in words_ncl[:2] if _double_metaphone_simple(t)]
                if codes:
                    phk = "|".join(codes)
                    if phk in target_phon_keys:
                        cnt_phon_key[phk] += 1
                        if len(idx_phon_key[phk]) < 60:
                            idx_phon_key[phk].append(ot_id)

            # 11. Country + token
            if cn and words_ncl:
                for t in words_ncl:
                    if len(t) >= 4 and t not in NAME_STOPWORDS:
                        pair = (cn, t)
                        if pair in target_country_toks:
                            cnt_country_tok[pair] += 1
                            if len(idx_country_tok[pair]) < 80:
                                idx_country_tok[pair].append(ot_id)

        del df_b
        gc.collect()

    scan_elapsed = time.time() - t0
    print(f"[{source_name}] Indexed in {scan_elapsed:.1f}s.")

    # Generate Candidate Pairs with Deterministic Ranking
    block_pairs = defaultdict(set)

    for s1_id, p in s1_profiles.items():
        # Pass 1: exact_name_norm
        if p["name_norm"] and cnt_name_norm[p["name_norm"]] <= 1000:
            for c in sorted(idx_name_norm[p["name_norm"]])[:50]:
                block_pairs["exact_name_norm"].add((s1_id, c))

        # Pass 2: exact_clean_legal
        if p["name_clean_legal"] and cnt_name_clean[p["name_clean_legal"]] <= 1000:
            for c in sorted(idx_name_clean[p["name_clean_legal"]])[:50]:
                block_pairs["exact_clean_legal"].add((s1_id, c))

        # Pass 3: rare_token_ranked
        if p["tokens_nn"]:
            tok_counts = Counter()
            for t in p["tokens_nn"]:
                if 1 <= cnt_rare_tokens[t] <= 500:
                    for ot_id in idx_rare_tokens[t]:
                        tok_counts[ot_id] += 1
            if tok_counts:
                ranked = sorted(tok_counts.items(), key=lambda x: (-x[1], x[0]))
                for ot_id, _ in ranked[:40]:
                    block_pairs["rare_token_ranked"].add((s1_id, ot_id))

        # Pass 4: address_num_distinctive_token (V3 NEW)
        if p["addr_num_words"]:
            addr_cands = Counter()
            for nw_key in p["addr_num_words"]:
                if 1 <= cnt_addr_nw[nw_key] <= 300:
                    for ot_id in idx_addr_nw[nw_key]:
                        addr_cands[ot_id] += 1
            if addr_cands:
                ranked = sorted(addr_cands.items(), key=lambda x: (-x[1], x[0]))
                for ot_id, _ in ranked[:35]:
                    block_pairs["address_num_distinctive_token"].add((s1_id, ot_id))

        # Pass 5: compact_name_prefix (V3 NEW)
        if p["compact_prefix"] and 1 <= cnt_compact[p["compact_prefix"]] <= 250:
            for c in sorted(idx_compact[p["compact_prefix"]])[:25]:
                block_pairs["compact_name_prefix"].add((s1_id, c))

        # Pass 6: rare_token_pairs (V3 NEW)
        if p["token_pairs"]:
            pair_cands = Counter()
            for tp in p["token_pairs"]:
                if 1 <= cnt_token_pairs[tp] <= 250:
                    for ot_id in idx_token_pairs[tp]:
                        pair_cands[ot_id] += 1
            if pair_cands:
                ranked = sorted(pair_cands.items(), key=lambda x: (-x[1], x[0]))
                for ot_id, _ in ranked[:30]:
                    block_pairs["rare_token_pairs"].add((s1_id, ot_id))

        # Pass 7: name_prefix_plus_addr_num (V3 NEW)
        if p["name_pfx_addr_num"]:
            pfx_num_cands = Counter()
            for pn in p["name_pfx_addr_num"]:
                if 1 <= cnt_pfx_num[pn] <= 150:
                    for ot_id in idx_pfx_num[pn]:
                        pfx_num_cands[ot_id] += 1
            if pfx_num_cands:
                ranked = sorted(pfx_num_cands.items(), key=lambda x: (-x[1], x[0]))
                for ot_id, _ in ranked[:25]:
                    block_pairs["name_prefix_plus_addr_num"].add((s1_id, ot_id))

        # Pass 8: first_token
        if p["first_tok"] and 1 <= cnt_first_tok[p["first_tok"]] <= 800:
            for c in sorted(idx_first_tok[p["first_tok"]])[:20]:
                block_pairs["first_token"].add((s1_id, c))

        # Pass 9: last_token
        if p["last_tok"] and 1 <= cnt_last_tok[p["last_tok"]] <= 400:
            for c in sorted(idx_last_tok[p["last_tok"]])[:15]:
                block_pairs["last_token"].add((s1_id, c))

        # Pass 10: sorted_token
        if p["sorted_key"] and cnt_sorted_key[p["sorted_key"]] <= 300:
            for c in sorted(idx_sorted_key[p["sorted_key"]])[:35]:
                block_pairs["sorted_token"].add((s1_id, c))

        # Pass 11: phonetic
        if p["phon_key"] and cnt_phon_key[p["phon_key"]] <= 200:
            for c in sorted(idx_phon_key[p["phon_key"]])[:25]:
                block_pairs["phonetic"].add((s1_id, c))

        # Pass 12: country_name_token
        if p["country_tokens"]:
            c_cands = Counter()
            for ct in p["country_tokens"]:
                if cnt_country_tok[ct] <= 250:
                    for ot_id in idx_country_tok[ct]:
                        c_cands[ot_id] += 1
            if c_cands:
                ranked = sorted(c_cands.items(), key=lambda x: (-x[1], x[0]))
                for ot_id, _ in ranked[:25]:
                    block_pairs["country_name_token"].add((s1_id, ot_id))

    return block_pairs


def run_v3_partitioned_tfidf(
    s1_df: pd.DataFrame,
    source_parquet_path: Path,
    min_sim: float = 0.45,
    top_k: int = 15,
) -> Set[Tuple[str, str]]:
    """Partitioned Char 3-Gram TF-IDF with min_sim=0.45 across full population."""
    t0 = time.time()
    s2_df = pd.read_parquet(source_parquet_path, columns=["entity_id", "name_clean_legal"])
    s2_df = s2_df[s2_df["name_clean_legal"] != ""].copy()
    s2_df["first_char"] = s2_df["name_clean_legal"].str[0].str.lower()

    sub_s1 = s1_df[s1_df["name_clean_legal"] != ""].copy()
    sub_s1["first_char"] = sub_s1["name_clean_legal"].str[0].str.lower()

    tfidf_pairs = set()
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
                tfidf_pairs.add((s1_ids_part[r], s2_ids_part[c_idx]))
        except Exception:
            continue

    print(f"Partitioned TF-IDF (min_sim={min_sim}) generated {len(tfidf_pairs):,} pairs in {time.time() - t0:.1f}s")
    del s2_df, sub_s1
    gc.collect()
    return tfidf_pairs


def evaluate_ablation(all_pass_pairs: dict, gt_pairs: set, val_s1: pd.DataFrame, source_label: str) -> dict:
    total_gt = len(gt_pairs)
    total_s1 = len(val_s1)

    print("\n" + "=" * 110)
    print(f"V3 BLOCKING ABLATION REPORT: S1 -> {source_label.upper()} (Total True Pairs: {total_gt:,})")
    print("=" * 110)
    print(f"{'BLOCKER':<32} {'CANDIDATES':>12} {'GT RECOVERED':>14} {'NEW GT':>10} {'MARGINAL REC':>14} {'CUMULATIVE REC':>16} {'CAND/NEW_GT':>10}")
    print("-" * 110)

    cumulative_pairs = set()
    cumulative_tp = set()
    ablation_stats = []

    for bname, pairs in all_pass_pairs.items():
        tp = pairs & gt_pairs
        new_tp = tp - cumulative_tp
        cumulative_tp.update(tp)
        cumulative_pairs.update(pairs)

        marginal_rec = len(new_tp) / total_gt * 100 if total_gt > 0 else 0
        cum_rec = len(cumulative_tp) / total_gt * 100 if total_gt > 0 else 0
        cost = len(pairs) / max(1, len(new_tp))

        print(f"{bname:<32} {len(pairs):>12,} {len(tp):>14,} {len(new_tp):>10,} {marginal_rec:>13.2f}% {cum_rec:>15.2f}% {cost:>10.1f}")
        ablation_stats.append({
            "blocker": bname,
            "candidates": len(pairs),
            "gt_recovered": len(tp),
            "new_gt": len(new_tp),
            "marginal_recall_pct": marginal_rec,
            "cumulative_recall_pct": cum_rec,
            "cand_per_new_gt": cost,
        })

    print("-" * 110)
    total_cands = len(cumulative_pairs)
    total_tp = len(cumulative_tp)
    final_recall = total_tp / total_gt * 100 if total_gt > 0 else 0

    print(f"{'TOTAL V3 UNION':<32} {total_cands:>12,} {total_tp:>14,} {total_tp:>10,} {'-':>14} {final_recall:>15.2f}% {total_cands/max(1, total_tp):>10.1f}")
    print("=" * 110)

    # Coverage summary
    s1_gt_counts = Counter()
    for s1_id, _ in gt_pairs:
        s1_gt_counts[s1_id] += 1
    s1_tp_counts = Counter()
    for s1_id, ot_id in cumulative_tp:
        s1_tp_counts[s1_id] += 1

    complete_cov = sum(1 for s1 in s1_gt_counts if s1_tp_counts[s1] == s1_gt_counts[s1])
    zero_cov = sum(1 for s1 in s1_gt_counts if s1_tp_counts[s1] == 0)
    n_gt_s1 = len(s1_gt_counts)

    print(f"\n{source_label} COVERAGE SUMMARY:")
    print(f"  V3 Candidate Recall        : {final_recall:.2f}% ({total_tp:,} / {total_gt:,})")
    print(f"  Total Candidates           : {total_cands:,} ({total_cands/total_s1:.1f} per S1)")
    print(f"  Complete S1 Entity Recall  : {complete_cov/n_gt_s1*100:.2f}% ({complete_cov:,} / {n_gt_s1:,})")
    print(f"  Zero True-Match Retrieval  : {zero_cov/n_gt_s1*100:.2f}% ({zero_cov:,} / {n_gt_s1:,})")

    return {
        "source": source_label,
        "total_gt": total_gt,
        "recovered_tp": total_tp,
        "candidate_recall_pct": final_recall,
        "total_candidates": total_cands,
        "candidates_per_s1": total_cands / total_s1,
        "complete_s1_coverage_pct": complete_cov / n_gt_s1 * 100,
        "zero_s1_coverage_pct": zero_cov / n_gt_s1 * 100,
        "ablation": ablation_stats,
    }


def main():
    print("=" * 80)
    print("PHASE 3 & 4: V3 BLOCKING BENCHMARK & ABLATION (10,000 S1 Validation Set)")
    print("=" * 80)

    val_s1 = pd.read_parquet(OUTPUT_DIR / "val_s1_10k.parquet")
    with open(OUTPUT_DIR / "val_gt_10k.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_s2 = {(s1, ot) for s1, olist in gt_data["s2"].items() for ot in olist}
    gt_s3 = {(s1, ot) for s1, olist in gt_data["s3"].items() for ot in olist}

    # 1. Evaluate Source 2
    s2_path = NORMALIZED_DIR / "s2_normalized.parquet"
    s2_blocks = run_v3_streaming_blocking(val_s1, s2_path, "Source 2")
    s2_tfidf = run_v3_partitioned_tfidf(val_s1, s2_path, min_sim=0.45, top_k=15)
    s2_blocks["partitioned_tfidf"] = s2_tfidf

    s2_report = evaluate_ablation(s2_blocks, gt_s2, val_s1, "Source 2")

    # 2. Evaluate Source 3
    s3_path = NORMALIZED_DIR / "s3_normalized.parquet"
    s3_blocks = run_v3_streaming_blocking(val_s1, s3_path, "Source 3")
    s3_tfidf = run_v3_partitioned_tfidf(val_s1, s3_path, min_sim=0.45, top_k=15)
    s3_blocks["partitioned_tfidf"] = s3_tfidf

    s3_report = evaluate_ablation(s3_blocks, gt_s3, val_s1, "Source 3")

    # Save summary report
    summary = {"s2": s2_report, "s3": s3_report}
    summary_path = OUTPUT_DIR / "v3_blocking_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved V3 blocking summary to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
