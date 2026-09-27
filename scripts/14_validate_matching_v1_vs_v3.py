"""
scripts/14_validate_matching_v1_vs_v3.py
=========================================
Minimal Downstream Matching Validation Experiment: V1 vs V3 Candidates.

Evaluates existing trained CatBoost matcher on the deterministic 10k S1 validation
benchmark using identical models, normalized data, entities, and thresholds.

Measures:
  A. Blocking recall (V1 vs V3)
  B. Conditional matcher accuracy (among entities with GT in candidates)
  C. End-to-end metrics: Macro Precision, Macro Recall, Macro F0.5, Macro F1
  D. Top-1, Top-3, Top-5 recall
  E. Failure decomposition:
     - Type A: True match absent from candidates (blocking failure)
     - Type B: True match present but ranked below an incorrect candidate (ranking inversion)
     - Type C: True match top-ranked but rejected by threshold (threshold failure)
  F. Downstream conversion efficiency:
     "Of the blocking improvement from V1 -> V3, how much reaches final matching?"
  G. Scoring throughput and runtime
"""

import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier

# Setup path and unbuffered stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.matching.features import (
    FEATURE_NAMES,
    compute_features_batch,
)
from business_entity_resolution.matching.evaluation import (
    compute_entity_level_metrics,
)
from business_entity_resolution.preprocessing.normalization import (
    load_normalized_or_compute,
)

OPERATIONAL_THRESHOLD = 0.9800000190734863
THRESHOLD_SWEEP_GRID = [0.85, 0.90, 0.92, 0.95, 0.98]
CHUNK_SIZE = 250_000

OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "matching_validation_v1_vs_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PATH DISCOVERY & MODEL LOADING
# ============================================================

def find_models() -> List[Path]:
    """Locate the trained CatBoost fold models."""
    env_dir = os.environ.get("BER_MODELS_DIR")
    candidate_dirs = [
        Path(env_dir) if env_dir else None,
        REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models",
        REPO_ROOT / "models",
        REPO_ROOT / "dataset" / "student_resource" / "outputs" / "matching" / "models",
        Path("/home/ec2-user/Amazon-ML-Hackathon/data/student_resource/outputs/matching/models"),
    ]
    for d in candidate_dirs:
        if d and d.is_dir():
            fold_models = sorted(list(d.glob("catboost_matcher_fold*.cbm")))
            if len(fold_models) == 5:
                print(f"Found 5 CatBoost ensemble models in: {d}")
                return fold_models
            single_model = list(d.glob("*.cbm"))
            if single_model:
                print(f"Found CatBoost model(s) in: {d}")
                return single_model
    return []


def find_validation_dir() -> Path:
    """Locate directory containing val_s1_10k.parquet and val_gt_10k.json."""
    candidates = [
        REPO_ROOT / "data" / "student_resource" / "outputs" / "v3_blocking",
        REPO_ROOT / "data" / "student_resource" / "outputs" / "validation",
        REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking_v2",
    ]
    for d in candidates:
        if d.is_dir() and (d / "val_s1_10k.parquet").exists() and (d / "val_gt_10k.json").exists():
            return d
    return candidates[0]


# ============================================================
# SCHEMA ADAPTER
# ============================================================

def adapt_candidates(df: pd.DataFrame, source_prefix: str) -> pd.DataFrame:
    """
    Adapts candidate dataframe to the standard schema:
    [s1_entity_id, candidate_entity_id, blocking_passes]
    """
    if df.empty:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])

    res = df.copy()

    # Column renaming
    col_map = {
        "s1_idx": "s1_entity_id",
        f"{source_prefix.lower()}_idx": "candidate_entity_id",
        "matched_entity_id": "candidate_entity_id",
        "entity_id_other": "candidate_entity_id",
    }
    res.rename(columns=col_map, inplace=True)

    # Ensure required columns exist
    if "s1_entity_id" not in res.columns or "candidate_entity_id" not in res.columns:
        raise ValueError(f"Unable to map candidate columns. Available: {list(res.columns)}")

    # Handle blocking_passes
    if "blocking_passes" not in res.columns or res["blocking_passes"].isnull().all():
        # Every V3 candidate was retrieved by at least one valid blocking pass.
        # Set to 'v3_blocker' so feature 21 (num_blocking_passes) evaluates to 1.0
        # rather than being unfairly penalized with 0.0.
        res["blocking_passes"] = "v3_blocker"
    else:
        res["blocking_passes"] = res["blocking_passes"].fillna("v3_blocker").astype(str)

    return res[["s1_entity_id", "candidate_entity_id", "blocking_passes"]].drop_duplicates()


# ============================================================
# BATCH SCORING
# ============================================================

def score_candidates_batch(
    cand_df: pd.DataFrame,
    s1_lookup: Dict[str, Tuple[str, str, str, str]],
    other_lookup: Dict[str, Tuple[str, str, str, str]],
    models: List[CatBoostClassifier],
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Computes features and average ensemble probabilities in memory-safe chunks."""
    n_pairs = len(cand_df)
    if n_pairs == 0:
        return np.array([], dtype=np.float32)

    probs = np.zeros(n_pairs, dtype=np.float32)

    for start_idx in range(0, n_pairs, chunk_size):
        end_idx = min(start_idx + chunk_size, n_pairs)
        chunk = cand_df.iloc[start_idx:end_idx]

        s1_ids = chunk["s1_entity_id"].values
        cand_ids = chunk["candidate_entity_id"].values
        bp_list = chunk["blocking_passes"].values
        chunk_len = len(chunk)

        s1_n = [""] * chunk_len
        s1_cl = [""] * chunk_len
        s1_ad = [""] * chunk_len
        s1_co = [""] * chunk_len

        ot_n = [""] * chunk_len
        ot_cl = [""] * chunk_len
        ot_ad = [""] * chunk_len
        ot_co = [""] * chunk_len

        for i in range(chunk_len):
            s1_d = s1_lookup.get(s1_ids[i])
            if s1_d:
                s1_n[i], s1_cl[i], s1_ad[i], s1_co[i] = s1_d
            ot_d = other_lookup.get(cand_ids[i])
            if ot_d:
                ot_n[i], ot_cl[i], ot_ad[i], ot_co[i] = ot_d

        X_batch = compute_features_batch(
            s1_names=s1_n, s1_clean=s1_cl, s1_addrs=s1_ad, s1_countries=s1_co,
            ot_names=ot_n, ot_clean=ot_cl, ot_addrs=ot_ad, ot_countries=ot_co,
            blocking_passes=bp_list,
        )

        prob_sum = np.zeros(chunk_len, dtype=np.float32)
        for m in models:
            prob_sum += m.predict_proba(X_batch)[:, 1].astype(np.float32)
        probs[start_idx:end_idx] = prob_sum / len(models)

        del X_batch, prob_sum, s1_n, s1_cl, s1_ad, s1_co, ot_n, ot_cl, ot_ad, ot_co
        gc.collect()

    return probs


# ============================================================
# COMPREHENSIVE MATCHING EVALUATION
# ============================================================

def evaluate_matching_pipeline(
    cand_df: pd.DataFrame,
    probs: np.ndarray,
    gt_dict: Dict[str, Set[str]],
    all_s1_ids: List[str],
    label: str,
    threshold: float = OPERATIONAL_THRESHOLD,
) -> Dict:
    """Computes all required diagnostic, ranking, and end-to-end metrics."""
    s1_all_set = set(all_s1_ids)
    total_s1 = len(all_s1_ids)

    # 1. Total Ground Truth
    gt_pairs_all = {(s1, ot) for s1, oset in gt_dict.items() for ot in oset if s1 in s1_all_set}
    total_gt_pairs = len(gt_pairs_all)
    s1_with_gt = {s1 for s1, oset in gt_dict.items() if len(oset) > 0 and s1 in s1_all_set}
    total_s1_with_gt = len(s1_with_gt)

    # 2. Blocking Recall
    cand_pairs_set = set(zip(cand_df["s1_entity_id"], cand_df["candidate_entity_id"]))
    gt_in_cands = cand_pairs_set & gt_pairs_all
    blocking_tp = len(gt_in_cands)
    blocking_recall = blocking_tp / total_gt_pairs if total_gt_pairs > 0 else 0.0

    # 3. Entity-level scores grouping for ranking & conditional metrics
    entity_cand_scores = defaultdict(list)
    for s1, ot, p in zip(cand_df["s1_entity_id"], cand_df["candidate_entity_id"], probs):
        entity_cand_scores[s1].append((ot, float(p)))

    # Sort each entity's candidates descending by score
    for s1 in entity_cand_scores:
        entity_cand_scores[s1].sort(key=lambda x: -x[1])

    # 4. Top-1, Top-3, Top-5 Recall
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    conditional_top1_correct = 0
    s1_with_gt_in_cands = 0

    for s1 in s1_with_gt:
        true_targets = gt_dict[s1]
        cands = entity_cand_scores.get(s1, [])
        cand_target_set = {c[0] for c in cands}

        has_gt_in_cands = len(true_targets & cand_target_set) > 0
        if has_gt_in_cands:
            s1_with_gt_in_cands += 1

        top1_targets = {cands[0][0]} if len(cands) >= 1 else set()
        top3_targets = {c[0] for c in cands[:3]}
        top5_targets = {c[0] for c in cands[:5]}

        if true_targets & top1_targets:
            top1_correct += 1
            if has_gt_in_cands:
                conditional_top1_correct += 1
        if true_targets & top3_targets:
            top3_correct += 1
        if true_targets & top5_targets:
            top5_correct += 1

    top1_recall = top1_correct / total_s1_with_gt if total_s1_with_gt > 0 else 0.0
    top3_recall = top3_correct / total_s1_with_gt if total_s1_with_gt > 0 else 0.0
    top5_recall = top5_correct / total_s1_with_gt if total_s1_with_gt > 0 else 0.0
    cond_top1_accuracy = (
        conditional_top1_correct / s1_with_gt_in_cands if s1_with_gt_in_cands > 0 else 0.0
    )

    # 5. Threshold-based Predictions & Macro Evaluation
    accepted_mask = probs >= threshold
    pred_dict = {s1: set() for s1 in all_s1_ids}
    for s1, ot in zip(cand_df["s1_entity_id"][accepted_mask], cand_df["candidate_entity_id"][accepted_mask]):
        pred_dict[s1].add(ot)

    macro_metrics = compute_entity_level_metrics(pred_dict, gt_dict, all_s1_ids=s1_all_set)
    pred_pairs_set = {(s1, ot) for s1, pset in pred_dict.items() for ot in pset}
    final_tp = len(pred_pairs_set & gt_pairs_all)
    final_fp = len(pred_pairs_set - gt_pairs_all)
    pair_precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) > 0 else 0.0

    # 6. Failure Mode Decomposition
    type_a_blocking_fn = 0       # True match absent from candidates
    type_b_ranking_inversion = 0 # True match present but ranked below false candidate
    type_c_threshold_fn = 0      # True match top-ranked but rejected by threshold

    ranking_failure_cases = []

    for s1 in s1_with_gt:
        true_targets = gt_dict[s1]
        cands = entity_cand_scores.get(s1, [])
        cand_dict = dict(cands)

        for true_id in true_targets:
            if true_id not in cand_dict:
                type_a_blocking_fn += 1
            else:
                true_prob = cand_dict[true_id]
                # Check top-ranked candidate for this entity
                top_cand_id, top_cand_prob = cands[0]
                if top_cand_id != true_id and top_cand_prob > true_prob:
                    type_b_ranking_inversion += 1
                    if len(ranking_failure_cases) < 10:
                        ranking_failure_cases.append({
                            "s1_id": s1,
                            "true_match_id": true_id,
                            "true_match_prob": round(true_prob, 4),
                            "false_candidate_id": top_cand_id,
                            "false_candidate_prob": round(top_cand_prob, 4),
                        })
                elif true_prob < threshold:
                    type_c_threshold_fn += 1

    return {
        "label": label,
        "threshold": threshold,
        "candidate_count": len(cand_df),
        "total_gt_pairs": total_gt_pairs,
        "blocking_recall": blocking_recall,
        "blocking_tp": blocking_tp,
        "cond_top1_accuracy": cond_top1_accuracy,
        "top1_recall": top1_recall,
        "top3_recall": top3_recall,
        "top5_recall": top5_recall,
        "macro_precision": macro_metrics["macro_precision"],
        "macro_recall": macro_metrics["macro_recall"],
        "macro_f05": macro_metrics["macro_f05"],
        "macro_f1": macro_metrics["macro_f1"],
        "pair_precision": pair_precision,
        "final_tp": final_tp,
        "final_fp": final_fp,
        "failure_modes": {
            "type_a_blocking_fn": type_a_blocking_fn,
            "type_b_ranking_inversion": type_b_ranking_inversion,
            "type_c_threshold_fn": type_c_threshold_fn,
        },
        "sample_ranking_failures": ranking_failure_cases,
    }


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def main():
    start_total = time.time()
    print("=" * 80)
    print("14: MINIMAL VALIDATION EXPERIMENT — V1 vs V3 MATCHING EVALUATION")
    print("=" * 80)

    # 1. Models
    model_paths = find_models()
    if not model_paths:
        print("\nERROR: No trained CatBoost models found.")
        print("Please ensure model files (*.cbm) are available in data/student_resource/outputs/matching/models/ or models/.")
        sys.exit(1)

    print(f"\nLoading {len(model_paths)} CatBoost model(s)...")
    models = []
    for mp in model_paths:
        m = CatBoostClassifier()
        m.load_model(str(mp))
        models.append(m)

    # 2. Validation Benchmark
    val_dir = find_validation_dir()
    print(f"\nValidation directory: {val_dir}")
    val_s1_path = val_dir / "val_s1_10k.parquet"
    val_gt_path = val_dir / "val_gt_10k.json"

    if not val_s1_path.exists() or not val_gt_path.exists():
        print(f"ERROR: Benchmark files missing in {val_dir}.")
        sys.exit(1)

    val_s1 = pd.read_parquet(val_s1_path)
    all_s1_ids = list(val_s1["entity_id"])
    val_s1_set = set(all_s1_ids)
    print(f"Loaded {len(all_s1_ids):,} validation S1 entities.")

    with open(val_gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Parse ground truth
    gt_dict_s2 = {s1: set(m if isinstance(m, list) else [m]) for s1, m in gt_data.get("s1_s2", gt_data.get("s2", {})).items()}
    gt_dict_s3 = {s1: set(m if isinstance(m, list) else [m]) for s1, m in gt_data.get("s1_s3", gt_data.get("s3", {})).items()}

    # 3. Lookup dictionaries
    print("\nBuilding normalized lookup dictionaries...")
    s1_lookup = {}
    for eid, nn, ncl, an, cn in zip(
        val_s1["entity_id"], val_s1["name_norm"], val_s1["name_clean_legal"],
        val_s1["address_norm"], val_s1["country_norm"]
    ):
        s1_lookup[eid] = (str(nn or ""), str(ncl or ""), str(an or ""), str(cn or ""))

    train_dir = REPO_ROOT / "data" / "student_resource" / "dataset" / "train"

    # Load S2/S3 normalized data
    cols = ["entity_id", "name_norm", "name_clean_legal", "address_norm", "country_norm"]
    _, s2_norm, s3_norm = load_normalized_or_compute(train_dir, REPO_ROOT, columns=cols)

    s2_lookup = {
        eid: (str(nn or ""), str(ncl or ""), str(an or ""), str(cn or ""))
        for eid, nn, ncl, an, cn in zip(s2_norm["entity_id"], s2_norm["name_norm"], s2_norm["name_clean_legal"], s2_norm["address_norm"], s2_norm["country_norm"])
    }
    s3_lookup = {
        eid: (str(nn or ""), str(ncl or ""), str(an or ""), str(cn or ""))
        for eid, nn, ncl, an, cn in zip(s3_norm["entity_id"], s3_norm["name_norm"], s3_norm["name_clean_legal"], s3_norm["address_norm"], s3_norm["country_norm"])
    }
    del s2_norm, s3_norm
    gc.collect()
    print(f"Lookups ready (S1: {len(s1_lookup):,}, S2: {len(s2_lookup):,}, S3: {len(s3_lookup):,})")

    # 4. Evaluate Sources
    results = {}

    for src_name, other_lookup, gt_dict in [("S2", s2_lookup, gt_dict_s2), ("S3", s3_lookup, gt_dict_s3)]:
        print(f"\n{'#' * 80}")
        print(f"# EVALUATING S1 -> {src_name}")
        print(f"{'#' * 80}")

        # Candidate paths
        v1_cand_path = val_dir / f"val_v1_s1_{src_name.lower()}_candidates.parquet"
        if not v1_cand_path.exists():
            v1_cand_path = REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking" / f"s1_{src_name.lower()}_candidates.parquet"

        v3_cand_path = val_dir / f"val_v3_s1_{src_name.lower()}_candidates.parquet"
        if not v3_cand_path.exists():
            v3_cand_path = REPO_ROOT / "data" / "student_resource" / "outputs" / "v3_blocking" / f"val_v3_s1_{src_name.lower()}_candidates.parquet"

        # Load & adapt V1
        print(f"\nLoading V1 candidates from: {v1_cand_path.name}")
        if v1_cand_path.exists():
            df_v1_raw = pd.read_parquet(v1_cand_path)
            # Filter to validation S1s if loaded from full parquet
            if len(df_v1_raw) > 500_000:
                df_v1_raw = df_v1_raw[df_v1_raw["s1_entity_id"].isin(val_s1_set)]
            cand_v1 = adapt_candidates(df_v1_raw, src_name)
            del df_v1_raw
        else:
            print(f"Warning: {v1_cand_path} not found.")
            cand_v1 = pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])

        # Load & adapt V3
        print(f"Loading V3 candidates from: {v3_cand_path.name}")
        if v3_cand_path.exists():
            df_v3_raw = pd.read_parquet(v3_cand_path)
            cand_v3 = adapt_candidates(df_v3_raw, src_name)
            del df_v3_raw
        else:
            print(f"Warning: {v3_cand_path} not found.")
            cand_v3 = pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])

        print(f"Candidate counts — V1: {len(cand_v1):,} pairs | V3: {len(cand_v3):,} pairs")

        # Score V1
        t0 = time.time()
        print(f"\nScoring V1 {src_name} candidates ({len(cand_v1):,} pairs)...")
        probs_v1 = score_candidates_batch(cand_v1, s1_lookup, other_lookup, models)
        t_v1_score = time.time() - t0
        rate_v1 = len(cand_v1) / t_v1_score if t_v1_score > 0 else 0
        print(f"V1 scored in {t_v1_score:.1f}s ({rate_v1:,.0f} pairs/sec)")

        # Score V3
        t0 = time.time()
        print(f"Scoring V3 {src_name} candidates ({len(cand_v3):,} pairs)...")
        probs_v3 = score_candidates_batch(cand_v3, s1_lookup, other_lookup, models)
        t_v3_score = time.time() - t0
        rate_v3 = len(cand_v3) / t_v3_score if t_v3_score > 0 else 0
        print(f"V3 scored in {t_v3_score:.1f}s ({rate_v3:,.0f} pairs/sec)")

        # Evaluate at operational threshold
        eval_v1 = evaluate_matching_pipeline(cand_v1, probs_v1, gt_dict, all_s1_ids, f"V1_{src_name}")
        eval_v3 = evaluate_matching_pipeline(cand_v3, probs_v3, gt_dict, all_s1_ids, f"V3_{src_name}")

        eval_v1["runtime_seconds"] = t_v1_score
        eval_v3["runtime_seconds"] = t_v3_score

        # Downstream conversion efficiency
        delta_blocking = eval_v3["blocking_recall"] - eval_v1["blocking_recall"]
        delta_matching = eval_v3["macro_recall"] - eval_v1["macro_recall"]
        conversion_rate = (delta_matching / delta_blocking * 100) if delta_blocking > 0 else 0.0

        # Print comparison table
        print(f"\n{'=' * 80}")
        print(f"RESULTS SUMMARY: S1 -> {src_name} (Threshold = {OPERATIONAL_THRESHOLD:.4f})")
        print("=" * 80)
        print(f"{'METRIC':<32} {'V1 CANDIDATES':>16} {'V3 CANDIDATES':>16} {'DELTA':>14}")
        print("-" * 80)
        print(f"{'Candidate Pairs Scored':<32} {eval_v1['candidate_count']:>16,} {eval_v3['candidate_count']:>16,} {eval_v3['candidate_count'] - eval_v1['candidate_count']:>+14,}")
        print(f"{'Scoring Runtime (s)':<32} {t_v1_score:>16.1f} {t_v3_score:>16.1f} {t_v3_score - t_v1_score:>+14.1f}")
        print(f"{'Blocking Recall':<32} {eval_v1['blocking_recall']*100:>15.2f}% {eval_v3['blocking_recall']*100:>15.2f}% {delta_blocking*100:>+13.2f}%")
        print(f"{'Top-1 Recall (Entity)':<32} {eval_v1['top1_recall']*100:>15.2f}% {eval_v3['top1_recall']*100:>15.2f}% {(eval_v3['top1_recall']-eval_v1['top1_recall'])*100:>+13.2f}%")
        print(f"{'Top-3 Recall (Entity)':<32} {eval_v1['top3_recall']*100:>15.2f}% {eval_v3['top3_recall']*100:>15.2f}% {(eval_v3['top3_recall']-eval_v1['top3_recall'])*100:>+13.2f}%")
        print(f"{'Top-5 Recall (Entity)':<32} {eval_v1['top5_recall']*100:>15.2f}% {eval_v3['top5_recall']*100:>15.2f}% {(eval_v3['top5_recall']-eval_v1['top5_recall'])*100:>+13.2f}%")
        print(f"{'Conditional Top-1 Accuracy':<32} {eval_v1['cond_top1_accuracy']*100:>15.2f}% {eval_v3['cond_top1_accuracy']*100:>15.2f}% {(eval_v3['cond_top1_accuracy']-eval_v1['cond_top1_accuracy'])*100:>+13.2f}%")
        print("-" * 80)
        print(f"{'Final Macro Precision':<32} {eval_v1['macro_precision']:>16.4f} {eval_v3['macro_precision']:>16.4f} {eval_v3['macro_precision'] - eval_v1['macro_precision']:>+14.4f}")
        print(f"{'Final Macro Recall':<32} {eval_v1['macro_recall']:>16.4f} {eval_v3['macro_recall']:>16.4f} {delta_matching:>+14.4f}")
        print(f"{'Final Macro F0.5 (PRIMARY)':<32} {eval_v1['macro_f05']:>16.4f} {eval_v3['macro_f05']:>16.4f} {eval_v3['macro_f05'] - eval_v1['macro_f05']:>+14.4f}")
        print(f"{'Final Macro F1':<32} {eval_v1['macro_f1']:>16.4f} {eval_v3['macro_f1']:>16.4f} {eval_v3['macro_f1'] - eval_v1['macro_f1']:>+14.4f}")
        print("-" * 80)
        print(f"Downstream Conversion Rate : {conversion_rate:.1f}% of blocking recall gain preserved in final match")

        # Failure modes table
        print(f"\n--- Failure Mode Decomposition (S1 -> {src_name}) ---")
        print(f"{'Failure Category':<45} {'V1 Pairs':>15} {'V3 Pairs':>15}")
        print("-" * 77)
        print(f"{'Type A: Match absent from candidates (Blocking FN)':<45} {eval_v1['failure_modes']['type_a_blocking_fn']:>15,} {eval_v3['failure_modes']['type_a_blocking_fn']:>15,}")
        print(f"{'Type B: Match present but ranked below false cand':<45} {eval_v1['failure_modes']['type_b_ranking_inversion']:>15,} {eval_v3['failure_modes']['type_b_ranking_inversion']:>15,}")
        print(f"{'Type C: Match top-ranked but rejected by thresh':<45} {eval_v1['failure_modes']['type_c_threshold_fn']:>15,} {eval_v3['failure_modes']['type_c_threshold_fn']:>15,}")
        print("=" * 80)

        # Threshold grid sweep for V3
        print(f"\n--- V3 Threshold Sensitivity (S1 -> {src_name}) ---")
        print(f"{'Threshold':<12} {'Precision':>12} {'Recall':>12} {'Macro F0.5':>14} {'Final TP':>12}")
        print("-" * 65)
        for th in THRESHOLD_SWEEP_GRID:
            th_eval = evaluate_matching_pipeline(cand_v3, probs_v3, gt_dict, all_s1_ids, f"V3_th_{th}", threshold=th)
            print(f"{th:<12.2f} {th_eval['macro_precision']:>12.4f} {th_eval['macro_recall']:>12.4f} {th_eval['macro_f05']:>14.4f} {th_eval['final_tp']:>12,}")

        results[src_name] = {
            "v1": eval_v1,
            "v3": eval_v3,
            "downstream_conversion_pct": conversion_rate,
        }

    # Save summary report
    report_file = OUTPUT_DIR / "matching_validation_v1_vs_v3_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_total
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT COMPLETE in {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Full results report saved to: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
