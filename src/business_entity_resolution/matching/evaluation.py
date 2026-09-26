"""
Evaluation and threshold optimization module for Business Entity Resolution.
Computes entity-level Macro F0.5 aligned with competition scoring.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


def compute_entity_level_metrics(
    predictions_by_s1: Dict[str, Set[str]],
    ground_truth_by_s1: Dict[str, Set[str]],
    all_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    Computes Macro Precision, Recall, and F0.5 across all S1 entities.

    Parameters
    ----------
    predictions_by_s1 : Dict[str, Set[str]]
        Mapping of s1_entity_id -> set of predicted matched entity IDs.
    ground_truth_by_s1 : Dict[str, Set[str]]
        Mapping of s1_entity_id -> set of true matched entity IDs.
    all_s1_ids : Set[str], optional
        Complete universe of S1 IDs to evaluate over. If None, uses union of keys.

    Returns
    -------
    Dict[str, float]
        Dictionary with macro_precision, macro_recall, macro_f05, macro_f1.
    """
    if all_s1_ids is None:
        all_s1_ids = set(ground_truth_by_s1.keys())

    precisions: List[float] = []
    recalls: List[float] = []
    f05_scores: List[float] = []
    f1_scores: List[float] = []

    for s1_id in all_s1_ids:
        true_set = ground_truth_by_s1.get(s1_id, set())
        pred_set = predictions_by_s1.get(s1_id, set())

        # Empty true set (singleton entity with no matches in other sources)
        if len(true_set) == 0:
            if len(pred_set) == 0:
                precisions.append(1.0)
                recalls.append(1.0)
                f05_scores.append(1.0)
                f1_scores.append(1.0)
            else:
                precisions.append(0.0)
                recalls.append(1.0)
                f05_scores.append(0.0)
                f1_scores.append(0.0)
            continue

        # Non-empty true set, but empty predictions
        if len(pred_set) == 0:
            precisions.append(1.0)
            recalls.append(0.0)
            f05_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        # Both non-empty
        tp = len(pred_set & true_set)
        p = tp / len(pred_set)
        r = tp / len(true_set)

        precisions.append(p)
        recalls.append(r)

        # F0.5: (1.25 * P * R) / (0.25 * P + R)
        denom_05 = 0.25 * p + r
        f05 = (1.25 * p * r) / denom_05 if denom_05 > 0.0 else 0.0
        f05_scores.append(f05)

        # F1: (2 * P * R) / (P + R)
        denom_1 = p + r
        f1 = (2.0 * p * r) / denom_1 if denom_1 > 0.0 else 0.0
        f1_scores.append(f1)

    return {
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f05": float(np.mean(f05_scores)),
        "macro_f1": float(np.mean(f1_scores)),
        "num_evaluated_entities": len(all_s1_ids),
    }


def sweep_thresholds(
    s1_ids: np.ndarray,
    candidate_ids: np.ndarray,
    probabilities: np.ndarray,
    ground_truth_by_s1: Dict[str, Set[str]],
    val_s1_ids: Optional[Set[str]] = None,
    thresholds: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Sweeps probability thresholds to find the threshold that maximizes Macro F0.5.

    Returns
    -------
    Tuple[pd.DataFrame, float]
        (results_dataframe, best_threshold)
    """
    if thresholds is None:
        thresholds = [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
            0.85, 0.90, 0.95,
        ]

    if val_s1_ids is None:
        val_s1_ids = set(np.unique(s1_ids))

    # Filter pairs to validation S1 IDs
    val_mask = np.isin(s1_ids, list(val_s1_ids)) if len(val_s1_ids) < len(np.unique(s1_ids)) else np.ones(len(s1_ids), dtype=bool)
    sub_s1 = s1_ids[val_mask]
    sub_cand = candidate_ids[val_mask]
    sub_prob = probabilities[val_mask]

    records = []
    best_f05 = -1.0
    best_threshold = 0.5

    for th in thresholds:
        pass_mask = sub_prob >= th
        pred_s1 = sub_s1[pass_mask]
        pred_cand = sub_cand[pass_mask]

        # Aggregate into mapping
        pred_dict = defaultdict(set)
        for s, c in zip(pred_s1, pred_cand):
            pred_dict[s].add(c)

        metrics = compute_entity_level_metrics(
            predictions_by_s1=pred_dict,
            ground_truth_by_s1=ground_truth_by_s1,
            all_s1_ids=val_s1_ids,
        )

        records.append({
            "threshold": th,
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f05": metrics["macro_f05"],
            "macro_f1": metrics["macro_f1"],
            "predicted_pairs_count": int(np.sum(pass_mask)),
        })

        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_threshold = th

    df_results = pd.DataFrame(records)
    return df_results, best_threshold
