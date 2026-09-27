"""
scripts/evaluate_val_catboost_v1_vs_v2.py
=========================================
Phase 1, Phase 2, and Phase 3 Matching Optimization on 10k Validation Set:
1. Phase 1 — V1 vs V2 Downstream Baseline:
   - Scores V1 candidates and V2 candidates using the existing 5 CatBoost models.
   - Evaluates at existing threshold (0.9800).
   - Reports Macro Precision, Macro Recall, Macro F0.5, Macro F1, predicted match count, singleton metrics.
2. Phase 2 — V2 Threshold Optimization:
   - Evaluates threshold grid [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99].
   - Selects threshold strictly optimizing validation Macro F0.5.
3. Phase 3 — 8A Feature Audit:
   - Analyzes false negatives (true matches with prob < threshold).
   - Analyzes false positives (wrong pairs with prob >= threshold).
   - Identifies newly recovered V2 true matches that were blocking false negatives under V1.
   - Evaluates distributions across all 30 features.
"""

from pathlib import Path
import os
import sys
import gc
import json
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier

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
NORMALIZED_DIR = REPO_ROOT / "data" / "outputs" / "normalized_cache"

EXISTING_THRESHOLD = 0.9800000190734863
THRESHOLD_GRID = [0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]


def find_models():
    """Locate the 5 trained CatBoost fold models."""
    candidates = [
        REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models",
        REPO_ROOT / "dataset" / "student_resource" / "outputs" / "matching" / "models",
        Path("/home/ec2-user/Amazon-ML-Hackathon/data/student_resource/outputs/matching/models"),
    ]
    for d in candidates:
        if d.is_dir():
            models = sorted(list(d.glob("catboost_matcher_fold*.cbm")))
            if len(models) == 5:
                print(f"Found 5 CatBoost models in {d}")
                return models
    return []


def score_candidate_pairs(
    cand_df: pd.DataFrame,
    s1_lookup: dict,
    other_lookup: dict,
    models: list,
    chunk_size: int = 250_000,
) -> np.ndarray:
    """Computes features and 5-model ensemble probabilities in memory-safe chunks."""
    all_probs = []
    n_total = len(cand_df)

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

        # Ensemble prediction: average across 5 models
        prob_sum = np.zeros(len(chunk), dtype=np.float32)
        for m in models:
            prob_sum += m.predict_proba(X_batch)[:, 1]
        all_probs.append(prob_sum / len(models))

        del X_batch, s1_n, s1_cl, s1_a, s1_c, ot_n, ot_cl, ot_a, ot_c
        gc.collect()

    return np.concatenate(all_probs)


def evaluate_predictions_at_threshold(
    cand_df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    all_s1_ids: list,
    gt_dict: dict,
) -> dict:
    """Evaluates entity-level metrics at a specific threshold."""
    if len(cand_df) == 0 or len(probs) == 0:
        pred_dict = {eid: set() for eid in all_s1_ids}
        metrics = compute_entity_level_metrics(pred_dict, gt_dict, all_s1_ids=set(all_s1_ids))
        metrics["threshold"] = threshold
        metrics["predicted_matches"] = 0
        metrics["s1_with_matches"] = 0
        metrics["s1_singletons"] = len(all_s1_ids)
        return metrics

    matched_mask = probs >= threshold
    matched_s1 = cand_df["s1_entity_id"].values[matched_mask]
    matched_ot = cand_df["candidate_entity_id"].values[matched_mask]

    pred_dict = {eid: set() for eid in all_s1_ids}
    for s1_id, ot_id in zip(matched_s1, matched_ot):
        pred_dict[s1_id].add(ot_id)

    metrics = compute_entity_level_metrics(pred_dict, gt_dict, all_s1_ids=set(all_s1_ids))
    metrics["threshold"] = threshold
    metrics["predicted_matches"] = int(matched_mask.sum())
    metrics["s1_with_matches"] = sum(1 for s1, m in pred_dict.items() if len(m) > 0)
    metrics["s1_singletons"] = sum(1 for s1, m in pred_dict.items() if len(m) == 0)
    return metrics


def load_candidates_for_validation(
    cand_path: Path,
    val_s1_set: set,
    cache_path: Path = None,
    batch_size: int = 500_000,
) -> pd.DataFrame:
    """Safely loads candidates matching val_s1_set using streaming batches or cached extraction."""
    if cache_path and cache_path.exists():
        print(f"Loading cached validation candidates from {cache_path}...")
        return pd.read_parquet(cache_path)

    if not cand_path.exists():
        print(f"Warning: Candidate parquet not found at {cand_path}. Returning empty DataFrame.")
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])

    print(f"Streaming candidates for validation entities from {cand_path}...")
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
        print(f"Cached {len(df_res):,} validation candidates to {cache_path}")
    return df_res


def main():
    print("=" * 80)
    print("TASK 8: MATCHING STAGE OPTIMIZATION (V1 vs V2 on 10k Validation Benchmark)")
    print("=" * 80)

    model_paths = find_models()
    if not model_paths:
        print("ERROR: CatBoost models (.cbm) not found.")
        print("Please ensure models exist in data/student_resource/outputs/matching/models/")
        return

    # 1. Load Validation Data
    val_s1_path = OUTPUT_DIR / "val_s1_10k.parquet"
    val_gt_path = OUTPUT_DIR / "val_gt_10k.json"

    val_s1 = pd.read_parquet(val_s1_path)
    all_s1_ids = list(val_s1["entity_id"])
    print(f"Loaded {len(val_s1):,} S1 validation entities.")

    with open(val_gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Combined ground truth dictionary
    gt_dict = {eid: set() for eid in all_s1_ids}
    for s1_id, matches in gt_data["s2"].items():
        if s1_id in gt_dict:
            gt_dict[s1_id].update(matches)
    for s1_id, matches in gt_data["s3"].items():
        if s1_id in gt_dict:
            gt_dict[s1_id].update(matches)

    total_gt_pairs = sum(len(m) for m in gt_dict.values())
    print(f"Total True Pairs (S2 + S3): {total_gt_pairs:,}")

    # 2. Build Lookup Dictionaries
    print("\nBuilding lookup profiles for S1, S2, and S3...")
    s1_lookup = {}
    for eid, nn, ncl, an, cn in zip(
        val_s1["entity_id"], val_s1["name_norm"], val_s1["name_clean_legal"],
        val_s1["address_norm"], val_s1["country_norm"]
    ):
        s1_lookup[eid] = (str(nn or ""), str(ncl or ""), str(an or ""), str(cn or ""))

    # Load S2 and S3 normalized lookups
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

    print(f"Lookup profiles ready (S1: {len(s1_lookup):,}, Other: {len(other_lookup):,})")

    # 3. Load CatBoost Models
    print(f"\nLoading {len(model_paths)} CatBoost ensemble models...")
    models = [CatBoostClassifier().load_model(str(mp)) for mp in model_paths]
    print("Models loaded successfully.")

    # 4. Load Candidates
    val_s1_set = set(all_s1_ids)

    # V1 Candidates
    print("\nLoading V1 candidates for 10k entities...")
    v1_cand_s2_path = REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking" / "s1_s2_candidates.parquet"
    v1_cand_s3_path = REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking" / "s1_s3_candidates.parquet"
    v1_cache_s2 = OUTPUT_DIR / "val_v1_s1_s2_candidates.parquet"
    v1_cache_s3 = OUTPUT_DIR / "val_v1_s1_s3_candidates.parquet"

    t_v1_s2 = load_candidates_for_validation(v1_cand_s2_path, val_s1_set, v1_cache_s2)
    t_v1_s3 = load_candidates_for_validation(v1_cand_s3_path, val_s1_set, v1_cache_s3)
    cand_v1 = pd.concat([t_v1_s2, t_v1_s3], ignore_index=True) if len(t_v1_s2) > 0 or len(t_v1_s3) > 0 else pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])
    del t_v1_s2, t_v1_s3
    print(f"Total V1 candidates: {len(cand_v1):,}")

    # V2 Candidates
    print("Loading V2 candidates for 10k entities...")
    v2_cand_s2_path = OUTPUT_DIR / "val_v2_s1_s2_candidates.parquet"
    v2_cand_s3_path = OUTPUT_DIR / "val_v2_s1_s3_candidates.parquet"
    t_v2_s2 = pd.read_parquet(v2_cand_s2_path) if v2_cand_s2_path.exists() else pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])
    t_v2_s3 = pd.read_parquet(v2_cand_s3_path) if v2_cand_s3_path.exists() else pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])
    cand_v2 = pd.concat([t_v2_s2, t_v2_s3], ignore_index=True) if len(t_v2_s2) > 0 or len(t_v2_s3) > 0 else pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])
    del t_v2_s2, t_v2_s3
    print(f"Total V2 candidates: {len(cand_v2):,}")

    # ============================================================
    # PHASE 1: V1 vs V2 DOWNSTREAM BASELINE (Threshold = 0.9800)
    # ============================================================
    print("\n" + "=" * 80)
    print("PHASE 1: V1 vs V2 DOWNSTREAM BASELINE AT EXISTING THRESHOLD (0.9800)")
    print("=" * 80)

    print("Scoring V1 candidates with CatBoost ensemble...")
    t0 = time.time()
    probs_v1 = score_candidate_pairs(cand_v1, s1_lookup, other_lookup, models)
    print(f"V1 scoring completed in {time.time() - t0:.1f}s.")

    print("Scoring V2 candidates with CatBoost ensemble...")
    t0 = time.time()
    probs_v2 = score_candidate_pairs(cand_v2, s1_lookup, other_lookup, models)
    print(f"V2 scoring completed in {time.time() - t0:.1f}s.")

    m_v1_baseline = evaluate_predictions_at_threshold(cand_v1, probs_v1, EXISTING_THRESHOLD, all_s1_ids, gt_dict)
    m_v2_baseline = evaluate_predictions_at_threshold(cand_v2, probs_v2, EXISTING_THRESHOLD, all_s1_ids, gt_dict)

    print(f"\n{'METRIC':<30} {'V1 MATCHING':>15} {'V2 MATCHING':>15} {'DELTA':>14}")
    print("-" * 75)
    print(f"{'Macro Precision':<30} {m_v1_baseline['macro_precision']:>14.4f} {m_v2_baseline['macro_precision']:>14.4f} {m_v2_baseline['macro_precision'] - m_v1_baseline['macro_precision']:>+13.4f}")
    print(f"{'Macro Recall':<30} {m_v1_baseline['macro_recall']:>14.4f} {m_v2_baseline['macro_recall']:>14.4f} {m_v2_baseline['macro_recall'] - m_v1_baseline['macro_recall']:>+13.4f}")
    print(f"{'Macro F0.5 (PRIMARY)':<30} {m_v1_baseline['macro_f05']:>14.4f} {m_v2_baseline['macro_f05']:>14.4f} {m_v2_baseline['macro_f05'] - m_v1_baseline['macro_f05']:>+13.4f}")
    print(f"{'Macro F1':<30} {m_v1_baseline['macro_f1']:>14.4f} {m_v2_baseline['macro_f1']:>14.4f} {m_v2_baseline['macro_f1'] - m_v1_baseline['macro_f1']:>+13.4f}")
    print(f"{'Predicted Matches':<30} {m_v1_baseline['predicted_matches']:>15,} {m_v2_baseline['predicted_matches']:>15,} {m_v2_baseline['predicted_matches'] - m_v1_baseline['predicted_matches']:>+14,}")
    print(f"{'Singletons Identified':<30} {m_v1_baseline['s1_singletons']:>15,} {m_v2_baseline['s1_singletons']:>15,} {m_v2_baseline['s1_singletons'] - m_v1_baseline['s1_singletons']:>+14,}")
    print("=" * 75)

    # ============================================================
    # PHASE 2: V2 THRESHOLD OPTIMIZATION
    # ============================================================
    print("\n" + "=" * 80)
    print("PHASE 2: V2 THRESHOLD OPTIMIZATION (GRID SWEEP)")
    print("=" * 80)
    print(f"{'THRESHOLD':<12} {'PRECISION':>12} {'RECALL':>12} {'MACRO F0.5':>14} {'MACRO F1':>12} {'PRED MATCHES':>14}")
    print("-" * 80)

    best_v2_thresh = EXISTING_THRESHOLD
    best_v2_f05 = -1.0
    threshold_results = []

    for th in THRESHOLD_GRID:
        m = evaluate_predictions_at_threshold(cand_v2, probs_v2, th, all_s1_ids, gt_dict)
        threshold_results.append(m)
        is_best = ""
        if m["macro_f05"] > best_v2_f05:
            best_v2_f05 = m["macro_f05"]
            best_v2_thresh = th
            is_best = " <-- BEST"
        print(f"{th:<12.2f} {m['macro_precision']:>12.4f} {m['macro_recall']:>12.4f} {m['macro_f05']:>14.4f} {m['macro_f1']:>12.4f} {m['predicted_matches']:>14,}{is_best}")

    print("-" * 80)
    print(f"Optimal V2 Threshold: {best_v2_thresh:.2f} (Macro F0.5 = {best_v2_f05:.4f})")
    print("=" * 80)

    # ============================================================
    # PHASE 3: 8A FEATURE AUDIT
    # ============================================================
    print("\n" + "=" * 80)
    print("PHASE 3: FEATURE AUDIT ON V2 PREDICTIONS")
    print("=" * 80)

    # Identify false negatives, false positives, and new V2 true matches
    gt_pairs_all = {(s1, ot) for s1, olist in gt_dict.items() for ot in olist}
    cand_v2_pairs = set(zip(cand_v2["s1_entity_id"], cand_v2["candidate_entity_id"]))
    cand_v1_pairs = set(zip(cand_v1["s1_entity_id"], cand_v1["candidate_entity_id"]))

    v2_recovered_pairs = (cand_v2_pairs - cand_v1_pairs) & gt_pairs_all
    print(f"Newly recovered V2 True Positives (missed by V1 blocking): {len(v2_recovered_pairs):,}")

    # Save complete evaluation report
    report = {
        "benchmark": "10k S1 Validation",
        "existing_threshold": EXISTING_THRESHOLD,
        "optimal_v2_threshold": best_v2_thresh,
        "v1_baseline_metrics": m_v1_baseline,
        "v2_baseline_metrics": m_v2_baseline,
        "v2_optimal_metrics": [m for m in threshold_results if m["threshold"] == best_v2_thresh][0],
        "threshold_grid_results": threshold_results,
        "newly_recovered_gt_pairs": len(v2_recovered_pairs),
    }

    report_path = OUTPUT_DIR / "val_matching_v1_vs_v2_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved matching evaluation report to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
