"""
scripts/evaluate_val_matcher_v1_v2_v3.py
========================================
Phase 5: End-to-End Validation Matcher Comparison (V1 vs V2 vs V3 Blocking)

Evaluates:
  Does better blocking actually improve downstream matching performance?
  Compares V1, V2, and V3 blocking candidates on the EXACT SAME 10,000 S1 validation benchmark.

Models:
  Supports 5-fold ensemble of:
   - LightGBM Booster models (models/lightgbm/lightgbm_matcher_fold1..5.txt) [Default]
   - CatBoost Classifier models (models/catboost_matcher_fold1..5.cbm)

Metrics Reported:
  1. Blocking Metrics:
     - Candidate Count
     - Ground-Truth Pairs Retrieved
     - Candidate Blocking Recall (%)
     - Candidate Precision (%)
     - Complete S1 Entity Recall (%)
     - Zero True-Match Retrieval (%)
  2. Matching Metrics (at optimal / baseline threshold):
     - Macro Precision
     - Macro Recall
     - Macro F0.5 (Primary Competition Metric)
     - Macro F1
     - Predicted Matches
     - End-to-End Recall (% of total GT pairs matched)
     - Matching Precision (% of predictions that are true GT)
     - S1 Singletons Correctly Identified

Output:
  - Prints clean side-by-side comparison table (V1 vs V2 vs V3).
  - Saves full report to: data/student_resource/outputs/validation/val_matcher_v1_v2_v3_report.json
"""

from pathlib import Path
import os
import sys
import gc
import json
import time
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.matching import (
    FEATURE_NAMES,
    compute_features_batch,
    compute_entity_level_metrics,
)

OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "validation"
MODELS_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models"
NORMALIZED_DIR = REPO_ROOT / "data" / "outputs" / "normalized_cache"


def load_lightgbm_models(models_dir: Path) -> list:
    """Loads 5 LightGBM fold models."""
    import lightgbm as lgb
    lgb_dir = models_dir / "lightgbm" if (models_dir / "lightgbm").is_dir() else models_dir
    models = []
    for fold in range(1, 6):
        p = lgb_dir / f"lightgbm_matcher_fold{fold}.txt"
        if not p.is_file():
            raise FileNotFoundError(f"Missing LightGBM model: {p}")
        models.append(lgb.Booster(model_file=str(p)))
    print(f"Loaded {len(models)} LightGBM models from {lgb_dir}")
    return models


def load_catboost_models(models_dir: Path) -> list:
    """Loads 5 CatBoost fold models."""
    from catboost import CatBoostClassifier
    models = []
    for fold in range(1, 6):
        p = models_dir / f"catboost_matcher_fold{fold}.cbm"
        if not p.is_file():
            raise FileNotFoundError(f"Missing CatBoost model: {p}")
        cb = CatBoostClassifier().load_model(str(p))
        models.append(cb)
    print(f"Loaded {len(models)} CatBoost models from {models_dir}")
    return models


def score_candidate_pairs(
    cand_df: pd.DataFrame,
    s1_lookup: dict,
    other_lookup: dict,
    models: list,
    model_type: str = "lightgbm",
    chunk_size: int = 250_000,
) -> np.ndarray:
    """Computes features and 5-model ensemble probabilities in memory-safe chunks."""
    all_probs = []
    n_total = len(cand_df)
    if n_total == 0:
        return np.array([], dtype=np.float32)

    for start_idx in range(0, n_total, chunk_size):
        chunk = cand_df.iloc[start_idx : start_idx + chunk_size]
        s1_ids = chunk["s1_entity_id"].values
        ot_ids = chunk["candidate_entity_id"].values
        bps = chunk["blocking_passes"].values if "blocking_passes" in chunk.columns else [""] * len(chunk)

        s1_n, s1_cl, s1_a, s1_c = [], [], [], []
        for eid in s1_ids:
            prof = s1_lookup.get(eid, ("", "", "", ""))
            s1_n.append(prof[0])
            s1_cl.append(prof[1])
            s1_a.append(prof[2])
            s1_c.append(prof[3])

        ot_n, ot_cl, ot_a, ot_c = [], [], [], []
        for eid in ot_ids:
            prof = other_lookup.get(eid, ("", "", "", ""))
            ot_n.append(prof[0])
            ot_cl.append(prof[1])
            ot_a.append(prof[2])
            ot_c.append(prof[3])

        X_batch = compute_features_batch(s1_n, s1_cl, s1_a, s1_c, ot_n, ot_cl, ot_a, ot_c, bps)

        prob_sum = np.zeros(len(chunk), dtype=np.float32)
        if model_type == "lightgbm":
            for m in models:
                prob_sum += m.predict(X_batch).astype(np.float32)
        else:
            for m in models:
                prob_sum += m.predict_proba(X_batch)[:, 1].astype(np.float32)

        all_probs.append(prob_sum / len(models))
        del X_batch, s1_n, s1_cl, s1_a, s1_c, ot_n, ot_cl, ot_a, ot_c
        gc.collect()

    return np.concatenate(all_probs)


def load_candidates_for_validation(
    cand_path: Path,
    val_s1_set: set,
    cache_path: Path = None,
    batch_size: int = 500_000,
) -> pd.DataFrame:
    """Safely loads candidates matching val_s1_set using streaming batches or cached extraction."""
    if cache_path and cache_path.exists():
        print(f"Loading cached candidates from {cache_path.name}...")
        return pd.read_parquet(cache_path)

    if not cand_path.exists():
        print(f"Warning: Candidate file not found at {cand_path}")
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])

    print(f"Streaming validation candidates from {cand_path.name}...")
    pf = pq.ParquetFile(str(cand_path))
    kept = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"]):
        df_b = batch.to_pandas()
        mask = np.fromiter((eid in val_s1_set for eid in df_b["s1_entity_id"].values), dtype=bool, count=len(df_b))
        subset = df_b[mask]
        if len(subset) > 0:
            kept.append(subset)

    df_res = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])
    if cache_path and len(df_res) > 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_parquet(cache_path, index=False)
        print(f"Cached {len(df_res):,} validation candidates to {cache_path.name}")
    return df_res


def evaluate_end_to_end(
    cand_df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    all_s1_ids: list,
    gt_dict: dict,
    total_gt_pairs: int,
) -> dict:
    """Computes both candidate blocking metrics and end-to-end matching metrics."""
    gt_pairs_all = {(s1, ot) for s1, olist in gt_dict.items() for ot in olist}

    # 1. Candidate blocking metrics
    cand_pairs = set(zip(cand_df["s1_entity_id"], cand_df["candidate_entity_id"]))
    cand_tp = cand_pairs & gt_pairs_all
    cand_recall = len(cand_tp) / total_gt_pairs * 100 if total_gt_pairs > 0 else 0.0
    cand_prec = len(cand_tp) / len(cand_df) * 100 if len(cand_df) > 0 else 0.0

    s1_gt_counts = Counter()
    for s1, ot in gt_pairs_all:
        s1_gt_counts[s1] += 1
    s1_tp_counts = Counter()
    for s1, ot in cand_tp:
        s1_tp_counts[s1] += 1

    all_s1_with_gt = set(s1_gt_counts.keys())
    complete_s1 = sum(1 for s1 in all_s1_with_gt if s1_tp_counts[s1] == s1_gt_counts[s1])
    zero_s1 = sum(1 for s1 in all_s1_with_gt if s1_tp_counts[s1] == 0)
    n_gt_s1 = len(all_s1_with_gt)

    complete_s1_pct = complete_s1 / n_gt_s1 * 100 if n_gt_s1 > 0 else 0.0
    zero_s1_pct = zero_s1 / n_gt_s1 * 100 if n_gt_s1 > 0 else 0.0

    # 2. Downstream matching predictions
    matched_mask = probs >= threshold
    matched_s1 = cand_df["s1_entity_id"].values[matched_mask]
    matched_ot = cand_df["candidate_entity_id"].values[matched_mask]

    pred_dict = {eid: set() for eid in all_s1_ids}
    for s1_id, ot_id in zip(matched_s1, matched_ot):
        pred_dict[s1_id].add(ot_id)

    metrics = compute_entity_level_metrics(pred_dict, gt_dict, all_s1_ids=set(all_s1_ids))

    pred_pairs = set(zip(matched_s1, matched_ot))
    pred_tp = pred_pairs & gt_pairs_all
    end_to_end_recall = len(pred_tp) / total_gt_pairs * 100 if total_gt_pairs > 0 else 0.0
    matching_precision = len(pred_tp) / len(pred_pairs) * 100 if len(pred_pairs) > 0 else 0.0

    return {
        "candidate_count": len(cand_df),
        "candidates_per_s1": len(cand_df) / len(all_s1_ids),
        "blocking_tp": len(cand_tp),
        "blocking_recall": cand_recall,
        "candidate_precision": cand_prec,
        "complete_s1_pct": complete_s1_pct,
        "zero_s1_pct": zero_s1_pct,
        "threshold": threshold,
        "predicted_matches": len(pred_pairs),
        "matching_tp": len(pred_tp),
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f05": metrics["macro_f05"],
        "macro_f1": metrics["macro_f1"],
        "end_to_end_recall": end_to_end_recall,
        "matching_precision": matching_precision,
        "s1_singletons": sum(1 for s1, m in pred_dict.items() if len(m) == 0),
        "s1_with_matches": sum(1 for s1, m in pred_dict.items() if len(m) > 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Validate End-to-End Matcher on V1 vs V2 vs V3")
    parser.add_argument("--model-type", type=str, default="lightgbm", choices=["lightgbm", "catboost"], help="Matcher model type")
    parser.add_argument("--threshold", type=float, default=None, help="Inference threshold")
    args = parser.parse_args()

    default_thresh = 0.70 if args.model_type == "lightgbm" else 0.98
    eval_thresh = args.threshold if args.threshold is not None else default_thresh

    print("=" * 80)
    print(f"PHASE 5: END-TO-END VALIDATION EVALUATION (V1 vs V2 vs V3 BLOCKING)")
    print(f"Matcher Model: {args.model_type.upper()} (Threshold: {eval_thresh:.2f})")
    print("=" * 80)

    # 1. Load Validation Data
    val_s1 = pd.read_parquet(OUTPUT_DIR / "val_s1_10k.parquet")
    all_s1_ids = list(val_s1["entity_id"])
    val_s1_set = set(all_s1_ids)

    with open(OUTPUT_DIR / "val_gt_10k.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_dict = {eid: set() for eid in all_s1_ids}
    for s1_id, matches in gt_data["s2"].items():
        if s1_id in gt_dict:
            gt_dict[s1_id].update(matches)
    for s1_id, matches in gt_data["s3"].items():
        if s1_id in gt_dict:
            gt_dict[s1_id].update(matches)

    total_gt_pairs = sum(len(m) for m in gt_dict.values())
    print(f"Validation Benchmark: {len(all_s1_ids):,} S1 Entities | {total_gt_pairs:,} Total Ground Truth Pairs")

    # 2. Build Profiles
    print("\nBuilding lookup profiles for S1, S2, and S3...")
    s1_lookup = {}
    for eid, nn, ncl, an, cn in zip(
        val_s1["entity_id"], val_s1["name_norm"], val_s1["name_clean_legal"],
        val_s1["address_norm"], val_s1["country_norm"]
    ):
        s1_lookup[eid] = (str(nn or ""), str(ncl or ""), str(an or ""), str(cn or ""))

    other_lookup = {}
    for src_file in ["s2_normalized.parquet", "s3_normalized.parquet"]:
        p = NORMALIZED_DIR / src_file
        if p.exists():
            df_src = pd.read_parquet(p, columns=["entity_id", "name_norm", "name_clean_legal", "address_norm", "country_norm"])
            for eid, nn, ncl, an, cn in zip(
                df_src["entity_id"], df_src["name_norm"], df_src["name_clean_legal"],
                df_src["address_norm"], df_src["country_norm"]
            ):
                other_lookup[eid] = (str(nn or ""), str(ncl or ""), str(an or ""), str(cn or ""))
            del df_src
            gc.collect()

    print(f"Lookup profiles ready (S1: {len(s1_lookup):,}, Targets: {len(other_lookup):,})")

    # 3. Load Models
    if args.model_type == "lightgbm":
        models = load_lightgbm_models(MODELS_DIR)
    else:
        models = load_catboost_models(MODELS_DIR)

    # 4. Load Candidates for V1, V2, V3
    print("\n--- Loading Candidate Sets ---")

    # V1
    v1_cand_s2_path = REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking" / "s1_s2_candidates.parquet"
    v1_cand_s3_path = REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking" / "s1_s3_candidates.parquet"
    v1_cache_s2 = OUTPUT_DIR / "val_v1_s1_s2_candidates.parquet"
    v1_cache_s3 = OUTPUT_DIR / "val_v1_s1_s3_candidates.parquet"
    t_v1_s2 = load_candidates_for_validation(v1_cand_s2_path, val_s1_set, v1_cache_s2)
    t_v1_s3 = load_candidates_for_validation(v1_cand_s3_path, val_s1_set, v1_cache_s3)
    cand_v1 = pd.concat([t_v1_s2, t_v1_s3], ignore_index=True)
    del t_v1_s2, t_v1_s3
    print(f"V1 Candidates: {len(cand_v1):,} pairs ({len(cand_v1) / len(all_s1_ids):.1f}/S1)")

    # V2
    v2_cand_s2_path = OUTPUT_DIR / "val_v2_s1_s2_candidates.parquet"
    v2_cand_s3_path = OUTPUT_DIR / "val_v2_s1_s3_candidates.parquet"
    t_v2_s2 = pd.read_parquet(v2_cand_s2_path)
    t_v2_s3 = pd.read_parquet(v2_cand_s3_path)
    cand_v2 = pd.concat([t_v2_s2, t_v2_s3], ignore_index=True)
    del t_v2_s2, t_v2_s3
    print(f"V2 Candidates: {len(cand_v2):,} pairs ({len(cand_v2) / len(all_s1_ids):.1f}/S1)")

    # V3
    v3_cand_s2_path = OUTPUT_DIR / "val_v3_s1_s2_candidates.parquet"
    v3_cand_s3_path = OUTPUT_DIR / "val_v3_s1_s3_candidates.parquet"
    t_v3_s2 = pd.read_parquet(v3_cand_s2_path)
    t_v3_s3 = pd.read_parquet(v3_cand_s3_path)
    cand_v3 = pd.concat([t_v3_s2, t_v3_s3], ignore_index=True)
    del t_v3_s2, t_v3_s3
    print(f"V3 Candidates: {len(cand_v3):,} pairs ({len(cand_v3) / len(all_s1_ids):.1f}/S1)")

    # 5. Score Candidates
    print(f"\n--- Scoring Candidate Sets with {args.model_type.upper()} Ensemble ---")
    t0 = time.time()
    print("Scoring V1 candidates...")
    probs_v1 = score_candidate_pairs(cand_v1, s1_lookup, other_lookup, models, args.model_type)
    print(f"V1 scored in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("Scoring V2 candidates...")
    probs_v2 = score_candidate_pairs(cand_v2, s1_lookup, other_lookup, models, args.model_type)
    print(f"V2 scored in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("Scoring V3 candidates...")
    probs_v3 = score_candidate_pairs(cand_v3, s1_lookup, other_lookup, models, args.model_type)
    print(f"V3 scored in {time.time() - t0:.1f}s")

    # 6. Evaluate End-to-End Metrics
    m_v1 = evaluate_end_to_end(cand_v1, probs_v1, eval_thresh, all_s1_ids, gt_dict, total_gt_pairs)
    m_v2 = evaluate_end_to_end(cand_v2, probs_v2, eval_thresh, all_s1_ids, gt_dict, total_gt_pairs)
    m_v3 = evaluate_end_to_end(cand_v3, probs_v3, eval_thresh, all_s1_ids, gt_dict, total_gt_pairs)

    # 7. Print Master Comparison Table
    print("\n" + "=" * 90)
    print(f"END-TO-END VALIDATION COMPARISON: V1 vs V2 vs V3 BLOCKING ({args.model_type.upper()} @ {eval_thresh:.2f})")
    print("=" * 90)
    print(f"{'METRIC':<32} {'V1 (OLD)':>16} {'V2 (INTERMEDIATE)':>18} {'V3 (NEW)':>16}")
    print("-" * 90)
    print(f"{'Candidate Count':<32} {m_v1['candidate_count']:>16,} {m_v2['candidate_count']:>18,} {m_v3['candidate_count']:>16,}")
    print(f"{'Candidates per S1':<32} {m_v1['candidates_per_s1']:>16.1f} {m_v2['candidates_per_s1']:>18.1f} {m_v3['candidates_per_s1']:>16.1f}")
    print(f"{'Blocking Recall (%)':<32} {m_v1['blocking_recall']:>15.2f}% {m_v2['blocking_recall']:>17.2f}% {m_v3['blocking_recall']:>15.2f}%")
    print(f"{'Candidate Precision (%)':<32} {m_v1['candidate_precision']:>15.2f}% {m_v2['candidate_precision']:>17.2f}% {m_v3['candidate_precision']:>15.2f}%")
    print(f"{'Complete S1 Recall (%)':<32} {m_v1['complete_s1_pct']:>15.2f}% {m_v2['complete_s1_pct']:>17.2f}% {m_v3['complete_s1_pct']:>15.2f}%")
    print(f"{'Zero True-Match Retrieval (%)':<32} {m_v1['zero_s1_pct']:>15.2f}% {m_v2['zero_s1_pct']:>17.2f}% {m_v3['zero_s1_pct']:>15.2f}%")
    print("-" * 90)
    print(f"{'Macro Precision':<32} {m_v1['macro_precision']:>16.4f} {m_v2['macro_precision']:>18.4f} {m_v3['macro_precision']:>16.4f}")
    print(f"{'Macro Recall':<32} {m_v1['macro_recall']:>16.4f} {m_v2['macro_recall']:>18.4f} {m_v3['macro_recall']:>16.4f}")
    print(f"{'Macro F0.5 (PRIMARY COMPETITION)':<32} {m_v1['macro_f05']:>16.4f} {m_v2['macro_f05']:>18.4f} {m_v3['macro_f05']:>16.4f}")
    print(f"{'Macro F1':<32} {m_v1['macro_f1']:>16.4f} {m_v2['macro_f1']:>18.4f} {m_v3['macro_f1']:>16.4f}")
    print(f"{'Predicted Matches':<32} {m_v1['predicted_matches']:>16,} {m_v2['predicted_matches']:>18,} {m_v3['predicted_matches']:>16,}")
    print(f"{'End-to-End Recall (%)':<32} {m_v1['end_to_end_recall']:>15.2f}% {m_v2['end_to_end_recall']:>17.2f}% {m_v3['end_to_end_recall']:>15.2f}%")
    print(f"{'Matching Precision (%)':<32} {m_v1['matching_precision']:>15.2f}% {m_v2['matching_precision']:>17.2f}% {m_v3['matching_precision']:>15.2f}%")
    print(f"{'S1 Singletons Identified':<32} {m_v1['s1_singletons']:>16,} {m_v2['s1_singletons']:>18,} {m_v3['s1_singletons']:>16,}")
    print("=" * 90)

    # 8. Save JSON Report
    report = {
        "benchmark": "10k S1 Validation",
        "model_type": args.model_type,
        "threshold": eval_thresh,
        "v1": m_v1,
        "v2": m_v2,
        "v3": m_v3,
    }
    report_path = OUTPUT_DIR / f"val_matcher_v1_v2_v3_{args.model_type}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved master report to: {report_path}")


if __name__ == "__main__":
    main()
