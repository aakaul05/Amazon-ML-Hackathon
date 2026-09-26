"""
business_entity_resolution.matching.deterministic
=================================================
Deterministic rule evaluation and matching logic.
"""

from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def build_target_index(df: pd.DataFrame, key_cols: List[str]) -> Dict[Any, List[str]]:
    """
    Constructs a fast lookup dictionary: key -> list of candidate entity_ids.
    Guards against empty strings: rows with any empty/NaN key component are excluded.
    """
    valid_df = df.dropna(subset=key_cols)
    for c in key_cols:
        valid_df = valid_df[valid_df[c] != ""]

    if len(valid_df) == 0:
        return {}

    if len(key_cols) == 1:
        return valid_df.groupby(key_cols[0])["entity_id"].apply(list).to_dict()
    else:
        return valid_df.groupby(key_cols)["entity_id"].apply(list).to_dict()


def predict_for_s1(
    s1_df: pd.DataFrame,
    key_cols: List[str],
    target_index: Dict[Any, List[str]],
) -> List[List[str]]:
    """
    Retrieves all matching IDs for each S1 entity given a deterministic rule index.
    Guards against empty keys in S1 (returns [] if any key column is empty).
    """
    if not target_index:
        return [[] for _ in range(len(s1_df))]

    if len(key_cols) == 1:
        s1_keys = s1_df[key_cols[0]].values
        valid_mask = (s1_df[key_cols[0]].notna() & (s1_df[key_cols[0]] != "")).values
    else:
        valid_mask = pd.Series(True, index=s1_df.index)
        for c in key_cols:
            valid_mask &= (s1_df[c].notna()) & (s1_df[c] != "")
        valid_mask = valid_mask.values
        s1_keys = list(zip(*(s1_df[c].values for c in key_cols)))

    preds: List[List[str]] = []
    for is_valid, k in zip(valid_mask, s1_keys):
        if is_valid and k in target_index:
            preds.append(target_index[k])
        else:
            preds.append([])

    return preds


def evaluate_predictions(
    s1_ids: List[str],
    preds: List[List[str]],
    truth_dict: Dict[str, Set[str]],
    rule_name: str,
    source_name: str,
) -> Dict[str, Any]:
    """
    Computes rigorous entity-level macro F0.5, Precision, Recall,
    Ambiguity statistics, Singleton false-positive rates, and multi-match counts.
    """
    n_entities = len(s1_ids)

    # 1. Ambiguity & candidate count metrics
    pred_counts = np.array([len(p) for p in preds], dtype=np.int32)
    s1_matched_mask = pred_counts > 0
    s1_matched_count = int(np.sum(s1_matched_mask))
    coverage_pct = (s1_matched_count / n_entities) * 100.0 if n_entities > 0 else 0.0

    ambiguous_mask = pred_counts > 1
    s1_ambiguous_count = int(np.sum(ambiguous_mask))
    ambiguous_pct_total = (s1_ambiguous_count / n_entities) * 100.0 if n_entities > 0 else 0.0
    ambiguous_pct_matched = (
        (s1_ambiguous_count / s1_matched_count * 100.0) if s1_matched_count > 0 else 0.0
    )
    max_candidates = int(np.max(pred_counts)) if len(pred_counts) > 0 else 0
    mean_candidates_matched = (
        float(np.mean(pred_counts[s1_matched_mask])) if s1_matched_count > 0 else 0.0
    )

    # 2. Entity-level ground truth comparison
    precisions = np.zeros(n_entities, dtype=np.float64)
    recalls = np.zeros(n_entities, dtype=np.float64)
    f05s = np.zeros(n_entities, dtype=np.float64)

    singleton_count = 0
    singleton_fp_count = 0
    exact_match_count = 0
    multi_match_truth_count = 0
    partially_correct_multi_count = 0
    completely_correct_multi_count = 0

    pos_precisions = []
    pos_recalls = []
    pos_f05s = []

    for i, s1_id in enumerate(s1_ids):
        pred_set = set(preds[i])
        true_set = truth_dict.get(s1_id, set())

        p_len = len(pred_set)
        t_len = len(true_set)

        if t_len == 0:
            singleton_count += 1
            if p_len == 0:
                precisions[i] = 1.0
                recalls[i] = 1.0
                f05s[i] = 1.0
                exact_match_count += 1
            else:
                singleton_fp_count += 1
                precisions[i] = 0.0
                recalls[i] = 0.0
                f05s[i] = 0.0
        else:
            if t_len > 1:
                multi_match_truth_count += 1

            if p_len == 0:
                precisions[i] = 0.0
                recalls[i] = 0.0
                f05s[i] = 0.0
                pos_precisions.append(0.0)
                pos_recalls.append(0.0)
                pos_f05s.append(0.0)
            else:
                tp = len(pred_set & true_set)
                p = tp / p_len
                r = tp / t_len
                precisions[i] = p
                recalls[i] = r

                if tp == 0:
                    f05 = 0.0
                else:
                    f05 = (1.25 * p * r) / (0.25 * p + r)

                f05s[i] = f05
                pos_precisions.append(p)
                pos_recalls.append(r)
                pos_f05s.append(f05)

                if pred_set == true_set:
                    exact_match_count += 1
                    if t_len > 1:
                        completely_correct_multi_count += 1
                elif t_len > 1 and tp > 0:
                    partially_correct_multi_count += 1

    macro_p = float(np.mean(precisions)) if n_entities > 0 else 0.0
    macro_r = float(np.mean(recalls)) if n_entities > 0 else 0.0
    macro_f05 = float(np.mean(f05s)) if n_entities > 0 else 0.0

    pos_macro_p = float(np.mean(pos_precisions)) if pos_precisions else 0.0
    pos_macro_r = float(np.mean(pos_recalls)) if pos_recalls else 0.0
    pos_macro_f05 = float(np.mean(pos_f05s)) if pos_f05s else 0.0

    singleton_fp_rate = (singleton_fp_count / singleton_count) if singleton_count > 0 else 0.0

    return {
        "Rule": rule_name,
        "Source": source_name,
        "S1_Entities": n_entities,
        "Coverage_Count": s1_matched_count,
        "Coverage_Pct": coverage_pct,
        "Ambiguous_Count": s1_ambiguous_count,
        "Ambiguous_Pct_Total": ambiguous_pct_total,
        "Ambiguous_Pct_Matched": ambiguous_pct_matched,
        "Max_Candidates": max_candidates,
        "Mean_Candidates_Matched": mean_candidates_matched,
        "Macro_Precision": macro_p,
        "Macro_Recall": macro_r,
        "Macro_F0.5": macro_f05,
        "Pos_Macro_Precision": pos_macro_p,
        "Pos_Macro_Recall": pos_macro_r,
        "Pos_Macro_F0.5": pos_macro_f05,
        "Singleton_Count": singleton_count,
        "Singleton_FP_Count": singleton_fp_count,
        "Singleton_FP_Rate": singleton_fp_rate,
        "Exact_Match_Count": exact_match_count,
        "Multi_Match_Truth_Count": multi_match_truth_count,
        "Partially_Correct_Multi": partially_correct_multi_count,
        "Completely_Correct_Multi": completely_correct_multi_count,
    }
