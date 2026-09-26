"""
business_entity_resolution.blocking.candidate_generator
========================================================
High-performance candidate generation and blocking algorithms for entity resolution.
"""

import time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn


def block_exact_field(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    field: str,
    block_name: str,
) -> List[Tuple[str, str, str]]:
    """
    Exact string equality blocking pass on specified column field via vectorized merge.
    """
    t0 = time.time()
    s1_valid = s1_df[s1_df[field] != ""][["entity_id", field]]
    other_valid = other_df[other_df[field] != ""][["entity_id", field]]

    merged = s1_valid.merge(other_valid, on=field, suffixes=("_s1", "_other"))
    pairs = [
        (r.entity_id_s1, r.entity_id_other, block_name)
        for r in merged.itertuples(index=False)
    ]

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(pairs):,} candidate pairs ({dt:.1f}s)")
    return pairs


def block_rare_tokens(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    max_df: int = 500,
    max_cand_per_s1: int = 50,
) -> List[Tuple[str, str, str]]:
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


def block_address_tokens(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
) -> List[Tuple[str, str, str]]:
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
    Captures character-level typos and spelling mutations with C++ sparse_dot_topn.
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

    # High-performance C++ sparse matrix multiplication with top-n filtering
    top_sim = awesome_cossim_topn(
        X_s1,
        X_other.T,
        ntop=top_k,
        lower_bound=min_sim,
        use_threads=True,
        n_jobs=4,
    )

    coo = top_sim.tocoo()
    s1_ids = s1_sub["entity_id"].values
    other_ids = other_sub["entity_id"].values

    pairs = [
        (s1_ids[r], other_ids[c], block_name)
        for r, c in zip(coo.row, coo.col)
    ]

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(pairs):,} candidate pairs (sim >= {min_sim}, {dt:.1f}s)")
    return pairs


def combine_blocks_and_evaluate(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    total_true_pairs: int,
    all_blocks: List[List[Tuple[str, str, str]]],
    source_name: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Combines candidates from multiple blocking passes, annotates pass provenance,
    and calculates comprehensive retrieval recall and coverage metrics.
    """
    print("\n" + "=" * 70)
    print(f"EVALUATING BLOCKING PASSES & UNION FOR {source_name}")
    print("=" * 70)

    # Evaluate individual block recall & count
    block_pair_sets: Dict[str, Set[Tuple[str, str]]] = {}

    for block_list in all_blocks:
        if not block_list:
            continue
        bname = block_list[0][2]
        pair_set = {(r[0], r[1]) for r in block_list}
        block_pair_sets[bname] = pair_set

        tp = sum(1 for s1_id, ot_id in pair_set if s1_id in gt_dict and ot_id in gt_dict[s1_id])
        recall = tp / total_true_pairs if total_true_pairs > 0 else 0.0
        avg_cand = len(pair_set) / len(s1_df)

        print(f"  {bname:<25s} | Recall: {recall*100:6.2f}% ({tp:>9,} TP) | Candidates: {len(pair_set):>10,} (avg {avg_cand:.2f}/S1)")

    # Union all blocks into a candidate DataFrame
    all_pairs = []
    for block_list in all_blocks:
        if block_list:
            all_pairs.append(pd.DataFrame(block_list, columns=["s1_entity_id", "candidate_entity_id", "block"]))

    if not all_pairs:
        cand_df = pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])
    else:
        raw = pd.concat(all_pairs, ignore_index=True)
        cand_df = (
            raw.groupby(["s1_entity_id", "candidate_entity_id"])["block"]
            .agg(lambda x: "|".join(sorted(set(x))))
            .reset_index()
            .rename(columns={"block": "blocking_passes"})
        )

    total_candidates = len(cand_df)

    # Unique contribution
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
    all_cand_pairs = set(zip(cand_df["s1_entity_id"], cand_df["candidate_entity_id"]))
    total_tp = sum(1 for (s1_id, ot_id) in all_cand_pairs if s1_id in gt_dict and ot_id in gt_dict[s1_id])
    union_recall = total_tp / total_true_pairs if total_true_pairs > 0 else 0.0

    retrieved_by_s1 = cand_df.groupby("s1_entity_id")["candidate_entity_id"].apply(set).to_dict()

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
    full_cov_pct = full_coverage / total_gt_s1 * 100 if total_gt_s1 > 0 else 0.0
    zero_cov_pct = zero_coverage / total_gt_s1 * 100 if total_gt_s1 > 0 else 0.0

    naive_search_space = float(len(s1_df)) * float(len(other_df))
    reduction_ratio = 1.0 - (total_candidates / naive_search_space) if naive_search_space > 0 else 0.0

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
