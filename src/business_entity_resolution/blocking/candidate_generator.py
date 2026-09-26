"""
business_entity_resolution.blocking.candidate_generator
========================================================
High-performance candidate generation and blocking algorithms.
Optimized for maximum recall with 32 GB RAM budget.

Blocking Passes (10 total):
 1. Exact name_norm
 2. Exact name_clean_legal
 3. Rare/Informative Name Tokens (inverted index)
 4. Address Component tokens (numeric/postal codes)
 5. TF-IDF char 3-gram (high threshold 0.70)
 6. TF-IDF char 3-gram (relaxed threshold 0.45) — catches fuzzy misses
 7. Sorted-Token Blocking — catches word reordering
 8. Name Prefix Blocking (first 6 chars) — catches abbreviations
 9. Country + Name-token compound key — cross-source by geography
10. Phonetic Blocking (Double Metaphone) — catches transliterations/spelling
"""

import gc
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn


# ============================================================
# Pass 1: Exact Field Matching
# ============================================================

def block_exact_field(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    field: str,
    block_name: str,
    max_key_df: int = 1000,
    max_cand_per_s1: int = 50,
) -> pd.DataFrame:
    """
    Exact string equality blocking pass on specified column field via vectorized merge.
    Guards against generic key explosions (DF <= max_key_df) and caps candidates per S1 entity.
    Returns lightweight 2-column DataFrame [s1_entity_id, candidate_entity_id].
    """
    t0 = time.time()
    s1_valid = s1_df[s1_df[field] != ""][["entity_id", field]]
    other_valid = other_df[other_df[field] != ""][["entity_id", field]]

    # Filter out over-frequent generic keys (e.g. generic words matching thousands of records)
    key_counts = other_valid.groupby(field).size()
    valid_keys = key_counts[key_counts <= max_key_df].index
    other_valid = other_valid[other_valid[field].isin(valid_keys)]
    s1_valid = s1_valid[s1_valid[field].isin(valid_keys)]

    merged = s1_valid.merge(other_valid, on=field, suffixes=("_s1", "_other"))
    merged = merged.groupby("entity_id_s1").head(max_cand_per_s1)

    cand_df = merged[["entity_id_s1", "entity_id_other"]].rename(
        columns={"entity_id_s1": "s1_entity_id", "entity_id_other": "candidate_entity_id"}
    )
    cand_df["block"] = block_name

    del merged, s1_valid, other_valid, key_counts, valid_keys
    gc.collect()

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(cand_df):,} candidate pairs ({dt:.1f}s)")
    return cand_df


# ============================================================
# Pass 3: Rare / Informative Token Blocking
# ============================================================

def block_rare_tokens(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    max_df: int = 500,
    max_cand_per_s1: int = 30,
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


# ============================================================
# Pass 4: Address Component Blocking
# ============================================================

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


# ============================================================
# Pass 5 / 6: TF-IDF Character N-Gram Blocking
# ============================================================

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


# ============================================================
# Pass 7: Sorted-Token Blocking (Word Reordering)
# ============================================================

def block_sorted_tokens(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    max_key_df: int = 500,
    max_cand_per_s1: int = 50,
) -> pd.DataFrame:
    """
    Sorted Token Blocking:
    Sorts tokens alphabetically in name_clean_legal to create order-invariant keys.
    Catches "Smith John Corp" ↔ "John Smith Corp" and similar word reorderings.
    """
    t0 = time.time()
    block_name = "sorted_token"

    s1_valid = s1_df[s1_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()
    other_valid = other_df[other_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()

    # Create sorted-token key (alphabetical ordering of tokens)
    s1_valid["sorted_key"] = s1_valid["name_clean_legal"].str.split().apply(
        lambda tokens: " ".join(sorted(tokens)) if isinstance(tokens, list) and len(tokens) >= 2 else ""
    )
    other_valid["sorted_key"] = other_valid["name_clean_legal"].str.split().apply(
        lambda tokens: " ".join(sorted(tokens)) if isinstance(tokens, list) and len(tokens) >= 2 else ""
    )

    # Remove empty and too-short keys
    s1_valid = s1_valid[s1_valid["sorted_key"].str.len() >= 4]
    other_valid = other_valid[other_valid["sorted_key"].str.len() >= 4]

    # Filter over-frequent keys
    key_counts = other_valid.groupby("sorted_key").size()
    valid_keys = key_counts[key_counts <= max_key_df].index
    other_valid = other_valid[other_valid["sorted_key"].isin(valid_keys)]
    s1_valid = s1_valid[s1_valid["sorted_key"].isin(valid_keys)]

    merged = s1_valid[["entity_id", "sorted_key"]].merge(
        other_valid[["entity_id", "sorted_key"]],
        on="sorted_key",
        suffixes=("_s1", "_other"),
    )
    merged = merged.groupby("entity_id_s1").head(max_cand_per_s1)

    cand_df = merged[["entity_id_s1", "entity_id_other"]].rename(
        columns={"entity_id_s1": "s1_entity_id", "entity_id_other": "candidate_entity_id"}
    )
    cand_df["block"] = block_name

    del merged, s1_valid, other_valid, key_counts, valid_keys
    gc.collect()

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(cand_df):,} candidate pairs ({dt:.1f}s)")
    return cand_df


# ============================================================
# Pass 8: Name Prefix Blocking (Abbreviations)
# ============================================================

def block_name_prefix(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    prefix_len: int = 6,
    max_key_df: int = 200,
    max_cand_per_s1: int = 30,
) -> pd.DataFrame:
    """
    Name Prefix Blocking:
    Matches entities sharing the first N characters of name_clean_legal.
    Catches abbreviations ("Internat" catches "International" ↔ "Intl").
    Only fires on keys with sufficient length and limited frequency.
    """
    t0 = time.time()
    block_name = f"name_prefix_{prefix_len}"

    s1_valid = s1_df[s1_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()
    other_valid = other_df[other_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()

    # Create prefix key
    s1_valid["prefix_key"] = s1_valid["name_clean_legal"].str[:prefix_len]
    other_valid["prefix_key"] = other_valid["name_clean_legal"].str[:prefix_len]

    # Filter keys that are too short (below prefix_len means name itself was short)
    s1_valid = s1_valid[s1_valid["prefix_key"].str.len() >= prefix_len]
    other_valid = other_valid[other_valid["prefix_key"].str.len() >= prefix_len]

    # Filter over-frequent keys
    key_counts = other_valid.groupby("prefix_key").size()
    valid_keys = key_counts[key_counts <= max_key_df].index
    other_valid = other_valid[other_valid["prefix_key"].isin(valid_keys)]
    s1_valid = s1_valid[s1_valid["prefix_key"].isin(valid_keys)]

    merged = s1_valid[["entity_id", "prefix_key"]].merge(
        other_valid[["entity_id", "prefix_key"]],
        on="prefix_key",
        suffixes=("_s1", "_other"),
    )
    merged = merged.groupby("entity_id_s1").head(max_cand_per_s1)

    cand_df = merged[["entity_id_s1", "entity_id_other"]].rename(
        columns={"entity_id_s1": "s1_entity_id", "entity_id_other": "candidate_entity_id"}
    )
    cand_df["block"] = block_name

    del merged, s1_valid, other_valid, key_counts, valid_keys
    gc.collect()

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(cand_df):,} candidate pairs ({dt:.1f}s)")
    return cand_df


# ============================================================
# Pass 9: Country + Name Token Compound Key
# ============================================================

def block_country_name_token(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    max_key_df: int = 300,
    max_cand_per_s1: int = 40,
    min_token_len: int = 4,
) -> pd.DataFrame:
    """
    Country + Name Token Compound Blocking:
    Creates compound keys of (country, informative_name_token) for cross-source matching.
    Catches entities in the same country sharing any distinctive name word.
    """
    t0 = time.time()
    block_name = "country_name_token"

    # Stopwords to exclude (common words that are not informative)
    STOP_TOKENS = {
        "the", "and", "for", "group", "services", "company", "international",
        "global", "solutions", "management", "systems", "technologies",
        "consulting", "enterprises", "partners", "associates", "holdings",
        "industries", "products", "national", "general", "american",
        "business", "financial", "capital", "investments", "properties",
        "construction", "development", "insurance", "marketing", "trading",
        "logistics", "communications", "engineering", "electric", "energy",
    }

    s1_valid = s1_df[(s1_df["name_clean_legal"] != "") & (s1_df["country_norm"] != "")].copy()
    other_valid = other_df[(other_df["name_clean_legal"] != "") & (other_df["country_norm"] != "")].copy()

    # Build inverted index: (country, token) -> list of other entity_ids
    inverted_index = defaultdict(list)
    token_counts = Counter()

    for ot_id, name, country in zip(
        other_valid["entity_id"],
        other_valid["name_clean_legal"],
        other_valid["country_norm"],
    ):
        for token in set(name.split()):
            if len(token) >= min_token_len and token not in STOP_TOKENS:
                key = (country, token)
                token_counts[key] += 1

    # Only keep rare compound keys
    valid_compound_keys = {k for k, cnt in token_counts.items() if cnt <= max_key_df}
    del token_counts

    for ot_id, name, country in zip(
        other_valid["entity_id"],
        other_valid["name_clean_legal"],
        other_valid["country_norm"],
    ):
        for token in set(name.split()):
            if len(token) >= min_token_len and token not in STOP_TOKENS:
                key = (country, token)
                if key in valid_compound_keys:
                    inverted_index[key].append(ot_id)

    # Lookup for S1
    s1_ids_list = []
    ot_ids_list = []

    for s1_id, name, country in zip(
        s1_valid["entity_id"],
        s1_valid["name_clean_legal"],
        s1_valid["country_norm"],
    ):
        cand_set = set()
        for token in set(name.split()):
            if len(token) >= min_token_len and token not in STOP_TOKENS:
                key = (country, token)
                if key in inverted_index:
                    cand_set.update(inverted_index[key])
                    if len(cand_set) >= max_cand_per_s1:
                        break
        for ot_id in list(cand_set)[:max_cand_per_s1]:
            s1_ids_list.append(s1_id)
            ot_ids_list.append(ot_id)

    del inverted_index, valid_compound_keys
    gc.collect()

    cand_df = pd.DataFrame({
        "s1_entity_id": s1_ids_list,
        "candidate_entity_id": ot_ids_list,
        "block": block_name,
    })

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(cand_df):,} candidate pairs ({dt:.1f}s)")
    return cand_df


# ============================================================
# Pass 10: Phonetic Blocking (Double Metaphone)
# ============================================================

def _double_metaphone_simple(word: str) -> str:
    """
    Simplified phonetic hash for blocking (Soundex-like, but slightly more discriminative).
    Maps consonant clusters to single characters, strips vowels except leading.
    Produces a compact phonetic code for each word.
    """
    if not word or len(word) < 2:
        return ""

    word = word.lower()
    # Keep first letter, then map consonants
    CONSONANT_MAP = {
        'b': 'B', 'c': 'K', 'd': 'T', 'f': 'F', 'g': 'K', 'h': '',
        'j': 'J', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'p': 'P',
        'q': 'K', 'r': 'R', 's': 'S', 't': 'T', 'v': 'F', 'w': '',
        'x': 'KS', 'y': '', 'z': 'S',
    }

    result = [word[0].upper()]
    prev = result[0]
    for ch in word[1:]:
        mapped = CONSONANT_MAP.get(ch, '')
        if mapped and mapped != prev:
            result.append(mapped)
            prev = mapped

    return "".join(result)[:6]  # Cap at 6 chars for blocking key


def block_phonetic(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    max_key_df: int = 300,
    max_cand_per_s1: int = 40,
) -> pd.DataFrame:
    """
    Phonetic Blocking:
    Uses simplified phonetic hashing on the first significant word of name_clean_legal.
    Catches spelling variations and transliterations:
      "Schneider" ↔ "Schnieder", "Mikhail" ↔ "Michael", etc.
    """
    t0 = time.time()
    block_name = "phonetic"

    s1_valid = s1_df[s1_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()
    other_valid = other_df[other_df["name_clean_legal"] != ""][["entity_id", "name_clean_legal"]].copy()

    def make_phonetic_key(name: str) -> str:
        tokens = name.split()
        # Use the first 2 tokens' phonetic codes as the blocking key
        codes = []
        for t in tokens[:2]:
            code = _double_metaphone_simple(t)
            if code:
                codes.append(code)
        return "|".join(codes) if codes else ""

    s1_valid["phon_key"] = s1_valid["name_clean_legal"].apply(make_phonetic_key)
    other_valid["phon_key"] = other_valid["name_clean_legal"].apply(make_phonetic_key)

    # Filter empty keys
    s1_valid = s1_valid[s1_valid["phon_key"] != ""]
    other_valid = other_valid[other_valid["phon_key"] != ""]

    # Filter over-frequent keys
    key_counts = other_valid.groupby("phon_key").size()
    valid_keys = key_counts[key_counts <= max_key_df].index
    other_valid = other_valid[other_valid["phon_key"].isin(valid_keys)]
    s1_valid = s1_valid[s1_valid["phon_key"].isin(valid_keys)]

    merged = s1_valid[["entity_id", "phon_key"]].merge(
        other_valid[["entity_id", "phon_key"]],
        on="phon_key",
        suffixes=("_s1", "_other"),
    )
    merged = merged.groupby("entity_id_s1").head(max_cand_per_s1)

    cand_df = merged[["entity_id_s1", "entity_id_other"]].rename(
        columns={"entity_id_s1": "s1_entity_id", "entity_id_other": "candidate_entity_id"}
    )
    cand_df["block"] = block_name

    del merged, s1_valid, other_valid, key_counts, valid_keys
    gc.collect()

    dt = time.time() - t0
    print(f"[{block_name}] Generated {len(cand_df):,} candidate pairs ({dt:.1f}s)")
    return cand_df


# ============================================================
# Combine and Evaluate (Updated for 10 passes)
# ============================================================

# All known block names and their bitmask positions
BLOCK_BITMASK = {
    "exact_name_norm":    1,
    "exact_clean_legal":  2,
    "rare_token":         4,
    "address_component":  8,
    "char_ngram_0.70":   16,
    "char_ngram_0.45":   32,
    "sorted_token":      64,
    "name_prefix_6":    128,
    "country_name_token": 256,
    "phonetic":         512,
}


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
    and calculates comprehensive retrieval recall and coverage metrics using 100% C-vectorized Pandas operations.
    Fully avoids Python tuple/set loops that consume multi-gigabytes of memory.
    """
    print("\n" + "=" * 70)
    print(f"EVALUATING BLOCKING PASSES & UNION FOR {source_name}")
    print("=" * 70)

    # 1. Build a flat, lightweight Ground Truth DataFrame ONCE for vectorized matching
    gt_pairs = [(s1_id, ot_id) for s1_id, tset in gt_dict.items() for ot_id in tset]
    gt_df = pd.DataFrame(gt_pairs, columns=["s1_entity_id", "candidate_entity_id"])
    del gt_pairs
    gc.collect()

    # 2. Vectorized Evaluation per individual block
    for bdf in block_dfs:
        if bdf.empty:
            continue
        bname = bdf["block"].iloc[0]

        # Deduplicate block pairs
        b_unique = bdf[["s1_entity_id", "candidate_entity_id"]].drop_duplicates()
        tp_match = b_unique.merge(gt_df, on=["s1_entity_id", "candidate_entity_id"])
        tp = len(tp_match)
        recall = tp / total_true_pairs if total_true_pairs > 0 else 0.0
        avg_cand = len(b_unique) / len(s1_df)

        print(f"  {bname:<25s} | Recall: {recall*100:6.2f}% ({tp:>9,} TP) | Candidates: {len(b_unique):>10,} (avg {avg_cand:.2f}/S1)")
        del b_unique, tp_match

    # 3. Concatenate unique pairs & annotate blocking passes via low-memory left-merges
    valid_dfs = [df for df in block_dfs if not df.empty]
    if not valid_dfs:
        cand_df = pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])
    else:
        # Build bitmask-to-string lookup dynamically from blocks present
        present_blocks = {}
        for df in valid_dfs:
            bname = df["block"].iloc[0]
            if bname in BLOCK_BITMASK:
                present_blocks[bname] = BLOCK_BITMASK[bname]

        max_bits = max(present_blocks.values()) * 2 if present_blocks else 32
        BITMASK_TO_STR = {}
        for i in range(1, max_bits):
            passes = []
            for bname, bit_val in sorted(present_blocks.items(), key=lambda x: x[1]):
                if i & bit_val:
                    passes.append(bname)
            if passes:
                BITMASK_TO_STR[i] = "|".join(passes)

        # Step 3a: Assign pass bitmasks and concatenate
        dfs_with_mask = []
        for df in valid_dfs:
            bname = df["block"].iloc[0]
            bit_val = BLOCK_BITMASK.get(bname, 0)
            if bit_val == 0:
                continue
            sub = df[["s1_entity_id", "candidate_entity_id"]].drop_duplicates().copy()
            sub["mask"] = np.uint16(bit_val)
            dfs_with_mask.append(sub)

        concat_df = pd.concat(dfs_with_mask, ignore_index=True)
        del dfs_with_mask
        gc.collect()

        # Step 3b: Vectorized GroupBy Sum (since bits are distinct powers of 2, sum equals bitwise OR)
        cand_df = concat_df.groupby(["s1_entity_id", "candidate_entity_id"], as_index=False)["mask"].sum()
        del concat_df
        gc.collect()

        # Step 3c: Translate bitmask uint16 to blocking_passes string
        cand_df["blocking_passes"] = cand_df["mask"].map(BITMASK_TO_STR)
        cand_df.drop(columns=["mask"], inplace=True)

    total_candidates = len(cand_df)

    # 4. Vectorized Union Evaluation
    cand_pairs_df = cand_df[["s1_entity_id", "candidate_entity_id"]]
    matched_union = cand_pairs_df.merge(gt_df, on=["s1_entity_id", "candidate_entity_id"])
    total_tp = len(matched_union)
    union_recall = total_tp / total_true_pairs if total_true_pairs > 0 else 0.0

    # 5. Vectorized Entity-Level Complete Coverage Calculation
    gt_counts = gt_df.groupby("s1_entity_id").size().rename("n_true")
    retrieved_tp_counts = matched_union.groupby("s1_entity_id").size().rename("n_retrieved_tp")
    del matched_union, cand_pairs_df
    gc.collect()

    # Reindex over all S1 entities present in Ground Truth
    cov_df = pd.concat([gt_counts, retrieved_tp_counts], axis=1).fillna(0)
    del gt_counts, retrieved_tp_counts
    gc.collect()

    full_coverage = int((cov_df["n_retrieved_tp"] == cov_df["n_true"]).sum())
    zero_coverage = int((cov_df["n_retrieved_tp"] == 0).sum())
    total_gt_s1 = len(gt_dict)
    del cov_df, gt_df
    gc.collect()

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
