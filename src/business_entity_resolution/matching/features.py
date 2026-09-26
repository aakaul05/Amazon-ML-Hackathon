"""
Pairwise feature computation module for Business Entity Resolution matching pipeline.
Computes 18 similarity and agreement features for candidate entity pairs.
"""

from typing import List, Sequence
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES: List[str] = [
    "name_norm_exact",
    "name_clean_exact",
    "name_jaro_winkler",
    "name_levenshtein_sim",
    "name_token_sort_sim",
    "name_token_set_sim",
    "name_partial_sim",
    "name_token_jaccard",
    "name_token_overlap",
    "name_len_diff",
    "name_token_count_diff",
    "addr_exact",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_levenshtein_sim",
    "addr_numeric_match",
    "country_exact",
    "num_blocking_passes",
]


def compute_features_batch(
    s1_names: Sequence[str],
    s1_clean: Sequence[str],
    s1_addrs: Sequence[str],
    s1_countries: Sequence[str],
    ot_names: Sequence[str],
    ot_clean: Sequence[str],
    ot_addrs: Sequence[str],
    ot_countries: Sequence[str],
    blocking_passes: Sequence[str],
) -> np.ndarray:
    """
    Computes 18 pairwise features for a batch of candidate pairs.

    Parameters
    ----------
    s1_names : Sequence[str]
        Source 1 normalized business names (name_norm)
    s1_clean : Sequence[str]
        Source 1 legal-cleaned business names (name_clean_legal)
    s1_addrs : Sequence[str]
        Source 1 normalized addresses (address_norm)
    s1_countries : Sequence[str]
        Source 1 normalized countries (country_norm)
    ot_names : Sequence[str]
        Other source (S2/S3) normalized business names
    ot_clean : Sequence[str]
        Other source legal-cleaned business names
    ot_addrs : Sequence[str]
        Other source normalized addresses
    ot_countries : Sequence[str]
        Other source normalized countries
    blocking_passes : Sequence[str]
        Pipe-delimited string of blocking passes that generated each pair

    Returns
    -------
    np.ndarray
        Array of shape (N, 18) with dtype float32 containing all computed features.
    """
    n = len(s1_names)
    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

    for i in range(n):
        sn = s1_names[i] if s1_names[i] is not None and isinstance(s1_names[i], str) else ""
        on = ot_names[i] if ot_names[i] is not None and isinstance(ot_names[i], str) else ""
        sc = s1_clean[i] if s1_clean[i] is not None and isinstance(s1_clean[i], str) else ""
        oc = ot_clean[i] if ot_clean[i] is not None and isinstance(ot_clean[i], str) else ""
        sa = s1_addrs[i] if s1_addrs[i] is not None and isinstance(s1_addrs[i], str) else ""
        oa = ot_addrs[i] if ot_addrs[i] is not None and isinstance(ot_addrs[i], str) else ""
        scountry = s1_countries[i] if s1_countries[i] is not None and isinstance(s1_countries[i], str) else ""
        ocountry = ot_countries[i] if ot_countries[i] is not None and isinstance(ot_countries[i], str) else ""
        bp = blocking_passes[i] if blocking_passes[i] is not None and isinstance(blocking_passes[i], str) else ""

        # Feature 0: name_norm_exact
        features[i, 0] = 1.0 if sn == on and sn else 0.0

        # Feature 1: name_clean_exact
        features[i, 1] = 1.0 if sc == oc and sc else 0.0

        # Features 2-6: String similarity on cleaned names
        if sc and oc:
            features[i, 2] = JaroWinkler.normalized_similarity(sc, oc)
            features[i, 3] = fuzz.ratio(sc, oc) / 100.0
            features[i, 4] = fuzz.token_sort_ratio(sc, oc) / 100.0
            features[i, 5] = fuzz.token_set_ratio(sc, oc) / 100.0
            features[i, 6] = fuzz.partial_ratio(sc, oc) / 100.0
        else:
            features[i, 2] = 0.0
            features[i, 3] = 0.0
            features[i, 4] = 0.0
            features[i, 5] = 0.0
            features[i, 6] = 0.0

        # Features 7-8: Token Jaccard & Overlap for name
        s_tok = set(sc.split()) if sc else set()
        o_tok = set(oc.split()) if oc else set()
        inter = len(s_tok & o_tok)
        union = len(s_tok | o_tok)
        features[i, 7] = inter / union if union > 0 else 0.0
        min_len = min(len(s_tok), len(o_tok))
        features[i, 8] = inter / min_len if min_len > 0 else 0.0

        # Features 9-10: Length diffs for name
        slen, olen = len(sc), len(oc)
        max_len = max(slen, olen)
        features[i, 9] = abs(slen - olen) / max_len if max_len > 0 else 0.0
        features[i, 10] = float(min(abs(len(s_tok) - len(o_tok)), 10))

        # Feature 11: addr_exact
        features[i, 11] = 1.0 if sa == oa and sa else 0.0

        # Features 12-14: Address token similarities & Levenshtein
        sa_tok = set(sa.split()) if sa else set()
        oa_tok = set(oa.split()) if oa else set()
        a_inter = len(sa_tok & oa_tok)
        a_union = len(sa_tok | oa_tok)
        features[i, 12] = a_inter / a_union if a_union > 0 else 0.0
        a_min = min(len(sa_tok), len(oa_tok))
        features[i, 13] = a_inter / a_min if a_min > 0 else 0.0
        features[i, 14] = fuzz.ratio(sa, oa) / 100.0 if sa and oa else 0.0

        # Feature 15: Numeric address token match
        s_num = {t for t in sa_tok if any(c.isdigit() for c in t)}
        o_num = {t for t in oa_tok if any(c.isdigit() for c in t)}
        n_inter = len(s_num & o_num)
        n_union = len(s_num | o_num)
        features[i, 15] = n_inter / n_union if n_union > 0 else 0.0

        # Feature 16: country_exact
        features[i, 16] = 1.0 if scountry == ocountry and scountry else 0.0

        # Feature 17: num_blocking_passes
        features[i, 17] = float(bp.count("|") + 1) if bp else 0.0

    return features
