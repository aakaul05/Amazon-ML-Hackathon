"""
Ground-truth pair extraction and labeling module for candidate pairs.
"""

from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd


def _get_s1_column(gt_file_path: Path) -> str:
    """
    Detect the Source 1 entity ID column name in the ground-truth TSV.

    Supports:
        source1_entity_id
        s1_entity_id
    """
    header = pd.read_csv(
        gt_file_path,
        sep="\t",
        nrows=0,
    ).columns

    if "source1_entity_id" in header:
        return "source1_entity_id"

    if "s1_entity_id" in header:
        return "s1_entity_id"

    raise ValueError(
        "Could not find Source 1 entity ID column. "
        f"Expected 'source1_entity_id' or 's1_entity_id'. "
        f"Found columns: {list(header)}"
    )


def load_ground_truth_pairs(
    gt_file_path: Path,
    target_prefix: str = None,
) -> Set[Tuple[str, str]]:
    """
    Load ground truth TSV and extract positive
    (s1_id, matched_id) pairs.

    Parameters
    ----------
    gt_file_path : Path
        Path to train_ground_truth.tsv.

    target_prefix : str, optional
        If provided, only matched IDs starting with this
        prefix are included, e.g. "S2-" or "S3-".

    Returns
    -------
    Set[Tuple[str, str]]
        Set of positive (S1, target) pairs.
    """

    s1_col = _get_s1_column(gt_file_path)

    df = pd.read_csv(
        gt_file_path,
        sep="\t",
        dtype={
            s1_col: str,
            "matched_entity_ids": str,
        },
        usecols=[
            s1_col,
            "matched_entity_ids",
        ],
    )

    positive_pairs: Set[Tuple[str, str]] = set()

    for s1_id, matches in zip(
        df[s1_col],
        df["matched_entity_ids"],
    ):
        if pd.isna(s1_id):
            continue

        if pd.isna(matches) or not str(matches).strip():
            continue

        s1_id = str(s1_id).strip()

        for match_id in str(matches).split(","):
            match_id = match_id.strip()

            if not match_id:
                continue

            if (
                target_prefix is not None
                and not match_id.startswith(target_prefix)
            ):
                continue

            positive_pairs.add(
                (s1_id, match_id)
            )

    return positive_pairs


def load_ground_truth_dict(
    gt_file_path: Path,
) -> Dict[str, Set[str]]:
    """
    Load ground truth as:

        S1 entity ID -> set of matched entity IDs

    Example:

        {
            "S1-965667": {
                "S2-681193310",
                "S2-743505751",
                "S3-775321672"
            }
        }

    Returns
    -------
    Dict[str, Set[str]]
        Mapping from each S1 entity to its matched entities.
    """

    s1_col = _get_s1_column(gt_file_path)

    df = pd.read_csv(
        gt_file_path,
        sep="\t",
        dtype={
            s1_col: str,
            "matched_entity_ids": str,
        },
        usecols=[
            s1_col,
            "matched_entity_ids",
        ],
    )

    gt_dict: Dict[str, Set[str]] = {}

    for s1_id, matches in zip(
        df[s1_col],
        df["matched_entity_ids"],
    ):
        if pd.isna(s1_id):
            continue

        s1_id = str(s1_id).strip()

        if (
            pd.isna(matches)
            or not str(matches).strip()
        ):
            gt_dict[s1_id] = set()
            continue

        gt_dict[s1_id] = {
            match_id.strip()
            for match_id in str(matches).split(",")
            if match_id.strip()
        }

    return gt_dict


def label_candidates_batch(
    s1_ids: np.ndarray,
    candidate_ids: np.ndarray,
    gt_set: Set[Tuple[str, str]],
) -> np.ndarray:
    """
    Label candidate pairs as:

        1 = positive / true match
        0 = negative / non-match

    Parameters
    ----------
    s1_ids : np.ndarray
        Array of S1 entity IDs.

    candidate_ids : np.ndarray
        Array of candidate entity IDs.

    gt_set : Set[Tuple[str, str]]
        Set containing known positive pairs.

    Returns
    -------
    np.ndarray
        int8 array containing binary labels.
    """

    labels = np.fromiter(
        (
            1
            if (str(s1), str(candidate)) in gt_set
            else 0
            for s1, candidate in zip(
                s1_ids,
                candidate_ids,
            )
        ),
        dtype=np.int8,
        count=len(s1_ids),
    )

    return labels