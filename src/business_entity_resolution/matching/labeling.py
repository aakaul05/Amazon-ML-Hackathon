"""
Ground-truth pair extraction and labeling module for candidate pairs.
"""

from pathlib import Path
from typing import Dict, Set, Tuple
import numpy as np
import pandas as pd


def load_ground_truth_pairs(
    gt_file_path: Path,
    target_prefix: str = None,
) -> Set[Tuple[str, str]]:
    """
    Loads ground truth TSV and extracts positive (s1_id, matched_id) pairs.

    Parameters
    ----------
    gt_file_path : Path
        Path to train_ground_truth.tsv
    target_prefix : str, optional
        If provided (e.g. 'S2' or 'S3'), only pairs where matched entity starts
        with this prefix will be included.

    Returns
    -------
    Set[Tuple[str, str]]
        Set of (s1_entity_id, matched_entity_id) positive pairs.
    """
    df = pd.read_csv(
        gt_file_path,
        sep="\t",
        dtype={"s1_entity_id": str, "matched_entity_ids": str},
        usecols=["s1_entity_id", "matched_entity_ids"],
    )

    positive_pairs: Set[Tuple[str, str]] = set()

    for s1_id, matches in zip(df["s1_entity_id"], df["matched_entity_ids"]):
        if pd.isna(matches) or not str(matches).strip():
            continue
        for match_id in str(matches).split(","):
            match_id = match_id.strip()
            if not match_id:
                continue
            if target_prefix is not None and not match_id.startswith(target_prefix):
                continue
            positive_pairs.add((s1_id, match_id))

    return positive_pairs


def load_ground_truth_dict(
    gt_file_path: Path,
) -> Dict[str, Set[str]]:
    """
    Loads ground truth as a dictionary mapping s1_entity_id -> set of matched entity IDs.
    """
    df = pd.read_csv(
        gt_file_path,
        sep="\t",
        dtype={"s1_entity_id": str, "matched_entity_ids": str},
        usecols=["s1_entity_id", "matched_entity_ids"],
    )

    gt_dict: Dict[str, Set[str]] = {}
    for s1_id, matches in zip(df["s1_entity_id"], df["matched_entity_ids"]):
        if pd.isna(matches) or not str(matches).strip():
            gt_dict[s1_id] = set()
            continue
        gt_dict[s1_id] = {m.strip() for m in str(matches).split(",") if m.strip()}

    return gt_dict


def label_candidates_batch(
    s1_ids: np.ndarray,
    candidate_ids: np.ndarray,
    gt_set: Set[Tuple[str, str]],
) -> np.ndarray:
    """
    Labels candidate pairs as 1 (positive/match) or 0 (negative/non-match).

    Parameters
    ----------
    s1_ids : np.ndarray
        Array of S1 entity IDs
    candidate_ids : np.ndarray
        Array of candidate entity IDs
    gt_set : Set[Tuple[str, str]]
        Set of known positive pairs

    Returns
    -------
    np.ndarray
        int8 array of shape (N,) with binary labels.
    """
    labels = np.fromiter(
        (1 if (s1, cand) in gt_set else 0 for s1, cand in zip(s1_ids, candidate_ids)),
        dtype=np.int8,
        count=len(s1_ids),
    )
    return labels
