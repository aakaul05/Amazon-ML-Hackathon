"""
business_entity_resolution.blocking.candidate_generator
========================================================
High-performance, memory-efficient candidate generation and blocking algorithms.
"""

import gc
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
) -> pd.DataFrame:
    """
    Exact string equality blocking pass on specified column field via vectorized merge.
    Returns lightweight 2-column DataFrame [s1_entity_id, candidate_entity_id].
    """
    t0 = time.time()
    s1_valid = s1_df[s1_df[field] != ""][["entity_id", field]]
    other_valid = other_df[other_df[field] != ""][["entity_id", field]]

    merged = s1_valid.merge(other_valid, on=field, suffixes=("_s1", "_other"))
    cand_df = merged[["entity_id_s1", "entity_id_other"]].rename(
        columns={"entity_id_s1": "s1_entity_id", "entity_id_other": "candidate_entity_id"}
    )
    cand_df["block"] = block_name

    del merged, s1_valid, other_valid
    gc.collect()

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(cand_df):,} candidate pairs ({dt:.1f}s)")
    return cand_df


def block_rare_tokens(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    max_df: int = 500,
    max_cand_per_s1: int = 50,
) -> pd.DataFrame:
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
    del token_counts

    # Build inverted index for other
    inverted_index = defaultdict(list)
    for ot_id, name in zip(other_names["entity_id"], other_names["name_norm"]):
        for token in set(name.split()):
            if token in valid_tokens:
                inverted_index[token].append(ot_id)

    s1_ids_list = []
    ot_ids_list = []
    s1_names = s1_df[s1_df["name_norm"] != ""]
    for s1_id, name in zip(s1_names["entity_id"], s1_names["name_norm"]):
        cand_set = set()
        for token in set(name.split()):
            if token in inverted_index:
                cand_set.update(inverted_index[token])
                if len(cand_set) >= max_cand_per_s1:
                    break
        for ot_id in list(cand_set)[:max_cand_per_s1]:
            s1_ids_list.append(s1_id)
            ot_ids_list.append(ot_id)

    del inverted_index, valid_tokens
    gc.collect()

    cand_df = pd.DataFrame({
        "s1_entity_id": s1_ids_list,
        "candidate_entity_id": ot_ids_list,
        "block": block_name
    })

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(cand_df):,} candidate pairs (DF <= {max_df}, {dt:.1f}s)")
    return cand_df


def block_address_tokens(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
) -> pd.DataFrame:
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
    del key_counts

    inverted_index = defaultdict(list)
    for ot_id, addr in zip(other_addrs["entity_id"], other_addrs["address_norm"]):
        for k in set(extract_address_keys(addr)):
            if k in valid_keys:
                inverted_index[k].append(ot_id)

    s1_ids_list = []
    ot_ids_list = []
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
            s1_ids_list.append(s1_id)
            ot_ids_list.append(ot_id)

    del inverted_index, valid_keys
    gc.collect()

    cand_df = pd.DataFrame({
        "s1_entity_id": s1_ids_list,
        "candidate_entity_id": ot_ids_list,
        "block": block_name
    })

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(cand_df):,} candidate pairs ({dt:.1f}s)")
    return cand_df


def block_tfidf_char_ngram(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    min_sim: float = 0.70,
    top_k: int = 10,
    sample_limit: int = 200000,
) -> pd.DataFrame:
    """
    TF-IDF Character 3-gram Nearest Neighbor Retrieval:
    Captures character-level typos and spelling mutations with C++ sparse_dot_topn.
    """
    t0 = time.time()
    block_name = "char_ngram"

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

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2)
    corpus = pd.concat([s1_sub["name_clean_legal"], other_sub["name_clean_legal"]])
    vectorizer.fit(corpus)

    X_s1 = vectorizer.transform(s1_sub["name_clean_legal"])
    X_other = vectorizer.transform(other_sub["name_clean_legal"])
    del corpus, vectorizer
    gc.collect()

    top_sim = awesome_cossim_topn(
        X_s1,
        X_other.T,
        ntop=top_k,
        lower_bound=min_sim,
        use_threads=True,
        n_jobs=4,
    )
    del X_s1, X_other
    gc.collect()

    coo = top_sim.tocoo()
    s1_ids = s1_sub["entity_id"].values
    other_ids = other_sub["entity_id"].values

    cand_df = pd.DataFrame({
        "s1_entity_id": s1_ids[coo.row],
        "candidate_entity_id": other_ids[coo.col],
        "block": block_name
    })

    del top_sim, coo, s1_ids, other_ids, s1_valid, other_valid, s1_sub, other_sub
    gc.collect()

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(cand_df):,} candidate pairs (sim >= {min_sim}, {dt:.1f}s)")
    return cand_df


def combine_blocks_and_evaluate(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    total_true_pairs: int,
    block_dfs: List[pd.DataFrame],
    source_name: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Combines candidates from multiple blocking passes, annotates pass provenance,
    and calculates comprehensive retrieval recall and coverage metrics with minimal memory footprint.
    """
    print("\n" + "=" * 70)
    print(f"EVALUATING BLOCKING PASSES & UNION FOR {source_name}")
    print("=" * 70)

    for bdf in block_dfs:
        if bdf.empty:
            continue
        bname = bdf["block"].iloc[0]
        pair_set = set(zip(bdf["s1_entity_id"], bdf["candidate_entity_id"]))
        tp = sum(1 for s1_id, ot_id in pair_set if s1_id in gt_dict and ot_id in gt_dict[s1_id])
        recall = tp / total_true_pairs if total_true_pairs > 0 else 0.0
        avg_cand = len(pair_set) / len(s1_df)

        print(f"  {bname:<25s} | Recall: {recall*100:6.2f}% ({tp:>9,} TP) | Candidates: {len(pair_set):>10,} (avg {avg_cand:.2f}/S1)")

    # Concatenate all block DataFrames
    valid_dfs = [df for df in block_dfs if not df.empty]
    if not valid_dfs:
        cand_df = pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])
    else:
        raw = pd.concat(valid_dfs, ignore_index=True)
        cand_df = (
            raw.groupby(["s1_entity_id", "candidate_entity_id"])["block"]
            .agg(lambda x: "|".join(sorted(set(x))))
            .reset_index()
            .rename(columns={"block": "blocking_passes"})
        )
        del raw
        gc.collect()

    total_candidates = len(cand_df)

    # Union Evaluation
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
