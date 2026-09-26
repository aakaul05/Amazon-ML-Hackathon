"""
Pairwise feature computation module for Business Entity Resolution matching pipeline.

Computes 30 similarity and agreement features for candidate entity pairs.
Features are designed to be discriminative WITHIN blocking candidates (where
name similarity is already high), not just across random pairs.

Key design principles:
  - Name features use RELATIVE differences (gaps from 1.0) rather than
    raw similarity, giving trees more variance to split on.
  - Cross-field features capture name-address interactions.
  - Structural features detect missing/mismatched fields.
  - Interaction features let the model combine signals non-linearly.
"""

from typing import List, Sequence
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein


FEATURE_NAMES: List[str] = [
    # --- Name similarity (raw) ---
    "name_norm_exact",          # 0
    "name_clean_exact",         # 1
    "name_jaro_winkler",        # 2
    "name_levenshtein_sim",     # 3
    "name_token_sort_sim",      # 4
    "name_token_set_sim",       # 5
    "name_partial_sim",         # 6
    "name_token_jaccard",       # 7
    "name_token_overlap",       # 8
    "name_len_diff",            # 9
    "name_token_count_diff",    # 10
    # --- Name GAPS from perfection (high variance within blocking) ---
    "name_jw_gap",              # 11: 1 - jaro_winkler  (bigger = worse match)
    "name_lev_gap",             # 12: 1 - levenshtein_sim
    "name_sort_gap",            # 13: 1 - token_sort_sim
    "name_edit_distance",       # 14: raw edit distance (absolute)
    # --- Address features ---
    "addr_exact",               # 15
    "addr_token_jaccard",       # 16
    "addr_token_overlap",       # 17
    "addr_levenshtein_sim",     # 18
    "addr_numeric_match",       # 19
    # --- Country ---
    "country_exact",            # 20
    # --- Blocking metadata ---
    "num_blocking_passes",      # 21
    # --- Cross-field & structural features (NEW) ---
    "name_addr_both_present",   # 22: both sides have non-empty name AND address
    "addr_missing_either",      # 23: 1 if either side has empty address
    "name_in_addr",             # 24: any name token appears in other side's address
    "common_rare_name_tokens",  # 25: count of shared name tokens ≥ 4 chars
    "name_containment",         # 26: is shorter name contained in longer name?
    "name_first_token_match",   # 27: first word of cleaned name matches
    # --- Interaction features (NEW) ---
    "name_x_addr_sim",          # 28: name_clean_sim * addr_sim (product interaction)
    "name_x_country",           # 29: name_clean_sim * country_exact
]

NUM_FEATURES = len(FEATURE_NAMES)


def _safe_str(val) -> str:
    """Coerce value to string, handling None/NaN."""
    if val is None or (isinstance(val, float) and val != val):
        return ""
    return str(val) if not isinstance(val, str) else val


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
    Computes 30 pairwise features for a batch of candidate pairs.

    Returns
    -------
    np.ndarray
        Array of shape (N, 30) with dtype float32.
    """
    n = len(s1_names)
    features = np.zeros((n, NUM_FEATURES), dtype=np.float32)

    for i in range(n):
        sn = _safe_str(s1_names[i])
        on = _safe_str(ot_names[i])
        sc = _safe_str(s1_clean[i])
        oc = _safe_str(ot_clean[i])
        sa = _safe_str(s1_addrs[i])
        oa = _safe_str(ot_addrs[i])
        scountry = _safe_str(s1_countries[i])
        ocountry = _safe_str(ot_countries[i])
        bp = _safe_str(blocking_passes[i])

        # ---- NAME RAW SIMILARITY (0-10) ----
        features[i, 0] = 1.0 if sn == on and sn else 0.0   # name_norm_exact
        features[i, 1] = 1.0 if sc == oc and sc else 0.0   # name_clean_exact

        jw_sim = 0.0
        lev_sim = 0.0
        sort_sim = 0.0
        set_sim = 0.0
        partial_sim = 0.0
        raw_edit = 0.0

        if sc and oc:
            jw_sim = JaroWinkler.normalized_similarity(sc, oc)
            lev_sim = fuzz.ratio(sc, oc) / 100.0
            sort_sim = fuzz.token_sort_ratio(sc, oc) / 100.0
            set_sim = fuzz.token_set_ratio(sc, oc) / 100.0
            partial_sim = fuzz.partial_ratio(sc, oc) / 100.0
            raw_edit = float(Levenshtein.distance(sc, oc))

        features[i, 2] = jw_sim
        features[i, 3] = lev_sim
        features[i, 4] = sort_sim
        features[i, 5] = set_sim
        features[i, 6] = partial_sim

        # Token Jaccard & Overlap
        s_tok = set(sc.split()) if sc else set()
        o_tok = set(oc.split()) if oc else set()
        inter = len(s_tok & o_tok)
        union_len = len(s_tok | o_tok)
        min_tok = min(len(s_tok), len(o_tok))

        features[i, 7] = inter / union_len if union_len > 0 else 0.0
        features[i, 8] = inter / min_tok if min_tok > 0 else 0.0

        # Length diffs
        slen, olen = len(sc), len(oc)
        max_len = max(slen, olen)
        features[i, 9] = abs(slen - olen) / max_len if max_len > 0 else 0.0
        features[i, 10] = float(min(abs(len(s_tok) - len(o_tok)), 10))

        # ---- NAME GAPS FROM PERFECTION (11-14) ----
        # These have MORE VARIANCE within blocking candidates than raw similarities
        features[i, 11] = 1.0 - jw_sim        # name_jw_gap
        features[i, 12] = 1.0 - lev_sim       # name_lev_gap
        features[i, 13] = 1.0 - sort_sim      # name_sort_gap
        features[i, 14] = raw_edit             # name_edit_distance

        # ---- ADDRESS (15-19) ----
        features[i, 15] = 1.0 if sa == oa and sa else 0.0   # addr_exact

        sa_tok = set(sa.split()) if sa else set()
        oa_tok = set(oa.split()) if oa else set()
        a_inter = len(sa_tok & oa_tok)
        a_union = len(sa_tok | oa_tok)
        a_min = min(len(sa_tok), len(oa_tok))

        addr_jaccard = a_inter / a_union if a_union > 0 else 0.0
        addr_overlap = a_inter / a_min if a_min > 0 else 0.0
        addr_lev = fuzz.ratio(sa, oa) / 100.0 if sa and oa else 0.0

        features[i, 16] = addr_jaccard
        features[i, 17] = addr_overlap
        features[i, 18] = addr_lev

        # Numeric address tokens
        s_num = {t for t in sa_tok if any(c.isdigit() for c in t)}
        o_num = {t for t in oa_tok if any(c.isdigit() for c in t)}
        n_inter = len(s_num & o_num)
        n_union = len(s_num | o_num)
        features[i, 19] = n_inter / n_union if n_union > 0 else 0.0

        # ---- COUNTRY (20) ----
        features[i, 20] = 1.0 if scountry == ocountry and scountry else 0.0

        # ---- BLOCKING PASSES (21) ----
        features[i, 21] = float(bp.count("|") + 1) if bp else 0.0

        # ---- CROSS-FIELD & STRUCTURAL (22-27) ----

        # 22: Both sides have name AND address
        features[i, 22] = 1.0 if (sc and oc and sa and oa) else 0.0

        # 23: Either side missing address
        features[i, 23] = 1.0 if (not sa or not oa) else 0.0

        # 24: Name tokens appearing in the other side's address
        name_in_addr_count = 0
        if s_tok and oa_tok:
            name_in_addr_count += len(s_tok & oa_tok)
        if o_tok and sa_tok:
            name_in_addr_count += len(o_tok & sa_tok)
        features[i, 24] = float(name_in_addr_count)

        # 25: Shared "rare" name tokens (length >= 4 chars, more likely meaningful)
        rare_s = {t for t in s_tok if len(t) >= 4}
        rare_o = {t for t in o_tok if len(t) >= 4}
        features[i, 25] = float(len(rare_s & rare_o))

        # 26: Name containment (is shorter name a substring of longer?)
        if sc and oc:
            shorter, longer = (sc, oc) if len(sc) <= len(oc) else (oc, sc)
            features[i, 26] = 1.0 if shorter in longer else 0.0
        else:
            features[i, 26] = 0.0

        # 27: First token of cleaned name matches
        if s_tok and o_tok:
            # Use sorted lists to get a stable "first" token
            s_first = sc.split()[0] if sc else ""
            o_first = oc.split()[0] if oc else ""
            features[i, 27] = 1.0 if s_first == o_first and s_first else 0.0
        else:
            features[i, 27] = 0.0

        # ---- INTERACTION FEATURES (28-29) ----
        # These let CatBoost use combined signals without requiring deeper trees.
        name_sim_for_interaction = sort_sim  # token_sort is order-invariant

        # 28: name_sim * addr_sim
        features[i, 28] = name_sim_for_interaction * addr_lev

        # 29: name_sim * country_exact
        features[i, 29] = name_sim_for_interaction * features[i, 20]

    return features
