"""
Script 08a: Entity-Based Matching Pilot (Task 8A).

Selects a random subset of S1 entities, loads ALL their candidate pairs
from Task 7 blocking, trains a CatBoost matcher, and evaluates with
proper separation of blocking failures vs. matcher failures.

Key design:
  - S1 entity selection → load ALL candidates for those entities
  - Blocking recall = GT pairs in candidates / total GT pairs
  - Matcher recall  = correctly predicted / GT pairs in candidates
  - End-to-end recall = correctly predicted / total GT pairs
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import KFold


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from business_entity_resolution.matching import (
    FEATURE_NAMES,
    compute_features_batch,
    label_candidates_batch,
    train_catboost_matcher,
    get_feature_importances,
)

from business_entity_resolution.preprocessing.normalization import (
    load_normalized_or_compute,
)


# -----------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------

def find_dataset_dir() -> Path:
    p = os.environ.get("BER_DATA_DIR")
    if p and Path(p).is_dir():
        return Path(p)
    for c in [
        REPO_ROOT / "dataset" / "student_resource" / "dataset" / "train",
        REPO_ROOT / "data" / "student_resource" / "dataset" / "train",
        Path.home() / "Amazon-ML-Hackathon" / "dataset" / "student_resource" / "dataset" / "train",
    ]:
        if c.is_dir():
            return c
    return REPO_ROOT / "dataset" / "student_resource" / "dataset" / "train"


def find_blocking_dir() -> Path:
    for c in [
        REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking",
        REPO_ROOT / "data" / "outputs" / "blocking",
        Path.home() / "Amazon-ML-Hackathon" / "data" / "student_resource" / "outputs" / "blocking",
    ]:
        if c.is_dir():
            return c
    return REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking"


OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "pilot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------
# Ground truth loader (using actual competition column names)
# -----------------------------------------------------------------

def load_ground_truth(
    gt_file_path: Path,
) -> Tuple[Dict[str, Set[str]], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    Returns:
        gt_dict:  S1 entity ID -> set of all matched entity IDs
        gt_pairs_s2: Set of (S1, S2) positive pairs
        gt_pairs_s3: Set of (S1, S3) positive pairs
    """
    df = pd.read_csv(
        gt_file_path, sep="\t",
        dtype={"source1_entity_id": str, "matched_entity_ids": str},
        usecols=["source1_entity_id", "matched_entity_ids"],
    )

    gt_dict: Dict[str, Set[str]] = {}
    gt_pairs_s2: Set[Tuple[str, str]] = set()
    gt_pairs_s3: Set[Tuple[str, str]] = set()

    for s1_id, matches in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        s1_id = str(s1_id)
        if pd.isna(matches) or not str(matches).strip():
            gt_dict[s1_id] = set()
            continue

        match_set = {m.strip() for m in str(matches).split(",") if m.strip()}
        gt_dict[s1_id] = match_set

        for mid in match_set:
            if mid.startswith("S2"):
                gt_pairs_s2.add((s1_id, mid))
            elif mid.startswith("S3"):
                gt_pairs_s3.add((s1_id, mid))

    return gt_dict, gt_pairs_s2, gt_pairs_s3


# -----------------------------------------------------------------
# Lookup table
# -----------------------------------------------------------------

def build_lookup_table(df_norm: pd.DataFrame) -> Dict[str, Tuple[str, str, str, str]]:
    lookup = {}
    for eid, nn, ncl, an, cn in zip(
        df_norm["entity_id"], df_norm["name_norm"],
        df_norm["name_clean_legal"], df_norm["address_norm"],
        df_norm["country_norm"],
    ):
        lookup[eid] = (
            nn if isinstance(nn, str) else "",
            ncl if isinstance(ncl, str) else "",
            an if isinstance(an, str) else "",
            cn if isinstance(cn, str) else "",
        )
    return lookup


# -----------------------------------------------------------------
# Memory-efficient candidate loading for selected S1 entities
# -----------------------------------------------------------------

def load_candidates_for_s1_entities(
    cand_path: Path,
    selected_s1_ids: Set[str],
    batch_size: int = 500_000,
) -> pd.DataFrame:
    """
    Streams through the candidate parquet in batches and keeps only
    rows whose s1_entity_id is in selected_s1_ids.
    Never loads the full parquet into RAM.
    """
    pf = pq.ParquetFile(str(cand_path))
    kept = []

    for batch in pf.iter_batches(
        batch_size=batch_size,
        columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"],
    ):
        df_b = batch.to_pandas()
        mask = np.fromiter(
            (eid in selected_s1_ids for eid in df_b["s1_entity_id"].values),
            dtype=bool, count=len(df_b),
        )
        subset = df_b[mask]
        if len(subset) > 0:
            kept.append(subset)

    if not kept:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "blocking_passes"])

    return pd.concat(kept, ignore_index=True)


# -----------------------------------------------------------------
# Entity-level threshold sweep with proper error separation
# -----------------------------------------------------------------

def entity_sweep_thresholds(
    s1_ids: np.ndarray,
    candidate_ids: np.ndarray,
    probabilities: np.ndarray,
    gt_in_candidates_by_s1: Dict[str, Set[str]],
    eval_s1_ids: Set[str],
    thresholds: List[float] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Sweeps thresholds using ONLY GT pairs present in the candidate set.
    This measures matcher quality without penalizing for blocking FNs.
    """
    if thresholds is None:
        thresholds = [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
            0.85, 0.90, 0.95,
        ]

    # Filter to eval S1 IDs
    eval_mask = np.fromiter(
        (eid in eval_s1_ids for eid in s1_ids),
        dtype=bool, count=len(s1_ids),
    )
    sub_s1 = s1_ids[eval_mask]
    sub_cand = candidate_ids[eval_mask]
    sub_prob = probabilities[eval_mask]

    records = []
    best_f05 = -1.0
    best_threshold = 0.5

    for th in thresholds:
        pass_mask = sub_prob >= th
        pred_s1 = sub_s1[pass_mask]
        pred_cand = sub_cand[pass_mask]

        # Aggregate predictions by S1
        pred_by_s1: Dict[str, Set[str]] = defaultdict(set)
        for s, c in zip(pred_s1, pred_cand):
            pred_by_s1[s].add(c)

        # Compute per-entity metrics against candidates-only GT
        precisions, recalls, f05_scores = [], [], []

        for s1_id in eval_s1_ids:
            true_in_cand = gt_in_candidates_by_s1.get(s1_id, set())
            predicted = pred_by_s1.get(s1_id, set())

            if len(true_in_cand) == 0:
                # No GT pairs survived blocking for this entity
                if len(predicted) == 0:
                    precisions.append(1.0)
                    recalls.append(1.0)
                    f05_scores.append(1.0)
                else:
                    precisions.append(0.0)
                    recalls.append(1.0)
                    f05_scores.append(0.0)
                continue

            if len(predicted) == 0:
                precisions.append(1.0)
                recalls.append(0.0)
                f05_scores.append(0.0)
                continue

            tp = len(predicted & true_in_cand)
            p = tp / len(predicted)
            r = tp / len(true_in_cand)

            precisions.append(p)
            recalls.append(r)

            denom = 0.25 * p + r
            f05 = (1.25 * p * r) / denom if denom > 0 else 0.0
            f05_scores.append(f05)

        macro_p = float(np.mean(precisions))
        macro_r = float(np.mean(recalls))
        macro_f05 = float(np.mean(f05_scores))

        records.append({
            "threshold": th,
            "macro_precision": macro_p,
            "cond_macro_recall": macro_r,
            "cond_macro_f05": macro_f05,
            "predicted_pairs": int(np.sum(pass_mask)),
        })

        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_threshold = th

    return pd.DataFrame(records), best_threshold


# -----------------------------------------------------------------
# Main pilot per source
# -----------------------------------------------------------------

def run_pilot_for_source(
    target_name: str,
    target_prefix: str,
    cand_path: Path,
    s1_lookup: Dict[str, Tuple[str, str, str, str]],
    target_lookup: Dict[str, Tuple[str, str, str, str]],
    gt_pairs: Set[Tuple[str, str]],
    gt_dict: Dict[str, Set[str]],
    n_s1_entities: int = 10_000,
):
    print("\n" + "=" * 70)
    print(f"ENTITY-BASED PILOT: S1 -> {target_name.upper()}")
    print(f"Selecting {n_s1_entities:,} random S1 entities")
    print("=" * 70)

    if not cand_path.is_file():
        print(f"ERROR: Candidate file not found: {cand_path}")
        return

    # ---------------------------------------------------------
    # Step 1: Select random S1 entities
    # ---------------------------------------------------------

    # Use S1 entities that have at least one GT match to this target
    s1_with_matches = {s1 for s1, _ in gt_pairs}
    # Also include some S1 entities with NO matches (singletons)
    all_s1 = list(s1_lookup.keys())
    s1_no_matches = [s for s in all_s1 if s not in s1_with_matches]

    rng = np.random.RandomState(42)
    n_with = min(int(n_s1_entities * 0.8), len(s1_with_matches))
    n_without = min(n_s1_entities - n_with, len(s1_no_matches))

    sampled_with = rng.choice(list(s1_with_matches), size=n_with, replace=False)
    sampled_without = rng.choice(s1_no_matches, size=n_without, replace=False) if n_without > 0 else []

    selected_s1 = set(sampled_with) | set(sampled_without)
    print(f"Selected {len(selected_s1):,} S1 entities "
          f"({len(sampled_with):,} with matches, {len(sampled_without):,} singletons)")

    # ---------------------------------------------------------
    # Step 2: Load ALL candidates for selected S1 entities
    # ---------------------------------------------------------

    print(f"Loading candidates for selected S1 entities (streaming)...")
    t0 = time.time()
    cand_df = load_candidates_for_s1_entities(cand_path, selected_s1)
    print(f"Loaded {len(cand_df):,} candidate pairs in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------
    # Step 3: Compute blocking diagnostics
    # ---------------------------------------------------------

    # GT pairs for selected S1 entities (to this target only)
    gt_for_selected: Dict[str, Set[str]] = {}
    total_gt_pairs = 0
    for s1_id in selected_s1:
        matches = gt_dict.get(s1_id, set())
        target_matches = {m for m in matches if m.startswith(target_prefix)}
        gt_for_selected[s1_id] = target_matches
        total_gt_pairs += len(target_matches)

    # GT pairs that appear in the candidate set
    cand_pair_set = set(zip(cand_df["s1_entity_id"], cand_df["candidate_entity_id"]))
    gt_in_candidates: Dict[str, Set[str]] = defaultdict(set)
    gt_recovered = 0
    for s1_id in selected_s1:
        for mid in gt_for_selected.get(s1_id, set()):
            if (s1_id, mid) in cand_pair_set:
                gt_in_candidates[s1_id].add(mid)
                gt_recovered += 1

    blocking_fn = total_gt_pairs - gt_recovered
    blocking_recall = gt_recovered / total_gt_pairs if total_gt_pairs > 0 else 0.0

    print(f"\n--- BLOCKING DIAGNOSTICS ---")
    print(f"S1 entities sampled      : {len(selected_s1):,}")
    print(f"Total candidate pairs    : {len(cand_df):,}")
    print(f"Total GT pairs (target)  : {total_gt_pairs:,}")
    print(f"GT pairs in candidates   : {gt_recovered:,}")
    print(f"Blocking false negatives : {blocking_fn:,}")
    print(f"Blocking recall          : {blocking_recall:.4f} ({blocking_recall*100:.2f}%)")

    del cand_pair_set
    gc.collect()

    # ---------------------------------------------------------
    # Step 4: Extract features for all candidate pairs
    # ---------------------------------------------------------

    print(f"\nExtracting features for {len(cand_df):,} pairs...")
    t0 = time.time()

    n_pairs = len(cand_df)
    s1_eids = cand_df["s1_entity_id"].values
    cand_eids = cand_df["candidate_entity_id"].values
    bp_list = cand_df["blocking_passes"].values

    s1_names = [""] * n_pairs
    s1_clean = [""] * n_pairs
    s1_addrs = [""] * n_pairs
    s1_countries = [""] * n_pairs
    ot_names = [""] * n_pairs
    ot_clean = [""] * n_pairs
    ot_addrs = [""] * n_pairs
    ot_countries = [""] * n_pairs

    for i in range(n_pairs):
        s1_data = s1_lookup.get(s1_eids[i])
        if s1_data is not None:
            s1_names[i], s1_clean[i], s1_addrs[i], s1_countries[i] = s1_data
        ot_data = target_lookup.get(cand_eids[i])
        if ot_data is not None:
            ot_names[i], ot_clean[i], ot_addrs[i], ot_countries[i] = ot_data

    print(f"Attribute extraction: {time.time() - t0:.1f}s")

    print(f"Computing {len(FEATURE_NAMES)} pairwise features...")
    t0 = time.time()
    features = compute_features_batch(
        s1_names=s1_names, s1_clean=s1_clean,
        s1_addrs=s1_addrs, s1_countries=s1_countries,
        ot_names=ot_names, ot_clean=ot_clean,
        ot_addrs=ot_addrs, ot_countries=ot_countries,
        blocking_passes=bp_list,
    )
    print(f"Feature computation: {time.time() - t0:.1f}s, shape={features.shape}")

    # ---------------------------------------------------------
    # Step 5: Label against GT
    # ---------------------------------------------------------

    # Build GT pair set for SELECTED entities only
    gt_pair_set = set()
    for s1_id in selected_s1:
        for mid in gt_for_selected.get(s1_id, set()):
            gt_pair_set.add((s1_id, mid))

    labels = label_candidates_batch(s1_eids, cand_eids, gt_pair_set)
    pos = int(np.sum(labels))
    neg = len(labels) - pos
    print(f"Labels: {pos:,} positives ({pos/len(labels)*100:.2f}%), {neg:,} negatives")

    # ---------------------------------------------------------
    # Step 6: 5-Fold Entity-Level Cross-Validation
    # ---------------------------------------------------------

    print("\nSetting up 5-Fold Entity-Level Cross-Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    selected_s1_sorted = np.array(sorted(list(selected_s1)))

    oof_probs = np.zeros(n_pairs, dtype=np.float32)
    feature_importances_list = []

    print(f"Starting 5-fold training and out-of-fold inference...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(selected_s1_sorted), 1):
        t_fold = time.time()
        fold_train_s1 = set(selected_s1_sorted[train_idx])
        fold_val_s1 = set(selected_s1_sorted[val_idx])

        train_mask = np.fromiter((eid in fold_train_s1 for eid in s1_eids), dtype=bool, count=n_pairs)
        val_mask = np.fromiter((eid in fold_val_s1 for eid in s1_eids), dtype=bool, count=n_pairs)

        X_tr, y_tr = features[train_mask], labels[train_mask]
        X_va, y_va = features[val_mask], labels[val_mask]

        print(f"\n--- Fold {fold}/5 ---")
        print(f"Train entities: {len(fold_train_s1):,} ({len(X_tr):,} pairs, {int(np.sum(y_tr)):,} pos)")
        print(f"Val entities  : {len(fold_val_s1):,} ({len(X_va):,} pairs, {int(np.sum(y_va)):,} pos)")

        model_path = OUTPUT_DIR / f"catboost_pilot_{target_name}_fold{fold}.cbm"
        model, metrics = train_catboost_matcher(
            X_train=X_tr, y_train=y_tr,
            X_val=X_va, y_val=y_va,
            iterations=1000, learning_rate=0.08, depth=7,
            thread_count=8, model_save_path=model_path, verbose=200,
            random_seed=42 + fold,
        )

        val_p = model.predict_proba(X_va)[:, 1]
        oof_probs[val_mask] = val_p

        fi_df = get_feature_importances(model, FEATURE_NAMES)
        feature_importances_list.append(fi_df)

        print(f"Fold {fold} finished in {time.time() - t_fold:.1f}s (best iter: {metrics['best_iteration']})")
        del model, X_tr, y_tr, X_va, y_va, train_mask, val_mask
        gc.collect()

    # Average feature importances across all 5 folds
    mean_fi = pd.concat(feature_importances_list).groupby("feature", as_index=False)["importance"].mean()
    mean_fi = mean_fi.sort_values(by="importance", ascending=False).reset_index(drop=True)
    print("\nTop 15 Mean Feature Importances Across 5 Folds:")
    print(mean_fi.head(15).to_string(index=False))
    mean_fi.to_csv(OUTPUT_DIR / f"feature_importance_{target_name}.csv", index=False)

    # ---------------------------------------------------------
    # Step 7: Threshold sweep on Out-Of-Fold (OOF) predictions
    # ---------------------------------------------------------

    print("\nSweeping thresholds on 5-Fold OOF predictions...")
    sweep_df, best_th = entity_sweep_thresholds(
        s1_ids=s1_eids,
        candidate_ids=cand_eids,
        probabilities=oof_probs,
        gt_in_candidates_by_s1=gt_in_candidates,
        eval_s1_ids=selected_s1,
    )

    print("\nThreshold Sweep (OOF Conditional on Candidates):")
    print(sweep_df.to_string(index=False))
    sweep_df.to_csv(OUTPUT_DIR / f"threshold_sweep_{target_name}.csv", index=False)

    # ---------------------------------------------------------
    # Step 8: Per-fold metrics evaluation at best OOF threshold
    # ---------------------------------------------------------

    fold_precisions = []
    fold_recalls = []
    fold_f05s = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(selected_s1_sorted), 1):
        fold_val_s1 = set(selected_s1_sorted[val_idx])
        fold_val_mask = np.fromiter((eid in fold_val_s1 for eid in s1_eids), dtype=bool, count=n_pairs)

        f_df, _ = entity_sweep_thresholds(
            s1_ids=s1_eids[fold_val_mask],
            candidate_ids=cand_eids[fold_val_mask],
            probabilities=oof_probs[fold_val_mask],
            gt_in_candidates_by_s1=gt_in_candidates,
            eval_s1_ids=fold_val_s1,
            thresholds=[best_th],
        )
        row = f_df.iloc[0]
        fold_precisions.append(row["macro_precision"])
        fold_recalls.append(row["cond_macro_recall"])
        fold_f05s.append(row["cond_macro_f05"])

    # ---------------------------------------------------------
    # Step 9: Detailed overall OOF metrics at best threshold
    # ---------------------------------------------------------

    best_row = sweep_df.loc[sweep_df["threshold"] == best_th].iloc[0]

    # Count matcher TP/FP/FN at best threshold
    pred_mask = oof_probs >= best_th
    pred_pairs = set(zip(s1_eids[pred_mask], cand_eids[pred_mask]))

    all_gt_in_cand_set = set()
    for s1_id in selected_s1:
        for mid in gt_in_candidates.get(s1_id, set()):
            all_gt_in_cand_set.add((s1_id, mid))

    all_gt_set = set()
    for s1_id in selected_s1:
        for mid in gt_for_selected.get(s1_id, set()):
            all_gt_set.add((s1_id, mid))

    matcher_tp = len(pred_pairs & all_gt_in_cand_set)
    matcher_fp = len(pred_pairs - all_gt_in_cand_set)
    matcher_fn = len(all_gt_in_cand_set - pred_pairs)
    blocking_fn_total = len(all_gt_set) - len(all_gt_in_cand_set)

    cond_recall = matcher_tp / len(all_gt_in_cand_set) if len(all_gt_in_cand_set) > 0 else 0.0
    e2e_recall = matcher_tp / len(all_gt_set) if len(all_gt_set) > 0 else 0.0
    pair_precision = matcher_tp / (matcher_tp + matcher_fp) if (matcher_tp + matcher_fp) > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"5-FOLD CV PILOT RESULTS: S1 -> {target_name.upper()}")
    print("=" * 70)
    print(f"  S1 entities evaluated  : {len(selected_s1):,}")
    print(f"  Total candidate pairs  : {len(cand_df):,}")
    print(f"  Total GT pairs        : {len(all_gt_set):,}")
    print(f"  GT pairs in cands     : {len(all_gt_in_cand_set):,}")
    print(f"  Blocking FN            : {blocking_fn_total:,}")
    print(f"  Blocking recall        : {len(all_gt_in_cand_set)/len(all_gt_set)*100:.2f}%" if len(all_gt_set) > 0 else "  Blocking recall: N/A")
    print(f"  ---")
    print(f"  Best OOF threshold     : {best_th:.2f}")
    print(f"  Predicted pairs        : {int(best_row['predicted_pairs']):,}")
    print(f"  Matcher TP             : {matcher_tp:,}")
    print(f"  Matcher FP             : {matcher_fp:,}")
    print(f"  Matcher FN (cond)      : {matcher_fn:,}")
    print(f"  ---")
    print(f"  Pair precision         : {pair_precision:.4f}")
    print(f"  Cond. matcher recall   : {cond_recall:.4f} ({cond_recall*100:.2f}%)")
    print(f"  End-to-end recall      : {e2e_recall:.4f} ({e2e_recall*100:.2f}%)")
    print(f"  Macro precision        : {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
    print(f"  Cond. macro recall     : {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
    print(f"  Cond. macro F0.5       : {np.mean(fold_f05s):.4f} ± {np.std(fold_f05s):.4f}")
    print("=" * 70)

    # Cleanup
    del cand_df, features, labels, oof_probs
    gc.collect()


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    train_dir = find_dataset_dir()
    blocking_dir = find_blocking_dir()

    print(f"Train Dataset Dir: {train_dir}")
    print(f"Blocking Outputs : {blocking_dir}")
    print(f"Pilot Outputs    : {OUTPUT_DIR}")

    # Load normalized cache
    print("\nLoading normalized cache...")
    columns = ["entity_id", "name_norm", "name_clean_legal", "address_norm", "country_norm"]
    s1_norm, s2_norm, s3_norm = load_normalized_or_compute(train_dir, REPO_ROOT, columns=columns)

    print("Building lookup dictionaries...")
    s1_lookup = build_lookup_table(s1_norm)
    s2_lookup = build_lookup_table(s2_norm)
    s3_lookup = build_lookup_table(s3_norm)
    del s1_norm, s2_norm, s3_norm
    gc.collect()

    # Load ground truth
    gt_file = train_dir / "train_ground_truth.tsv"
    if not gt_file.is_file():
        alt_gt = REPO_ROOT / "dataset" / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"
        if alt_gt.is_file():
            gt_file = alt_gt

    print(f"Loading ground truth from {gt_file}...")
    gt_dict, gt_pairs_s2, gt_pairs_s3 = load_ground_truth(gt_file)
    print(f"Loaded {len(gt_pairs_s2):,} S2 positive pairs, {len(gt_pairs_s3):,} S3 positive pairs.")

    # Run pilot for S2
    run_pilot_for_source(
        "s2", "S2",
        blocking_dir / "s1_s2_candidates.parquet",
        s1_lookup, s2_lookup,
        gt_pairs_s2, gt_dict,
        n_s1_entities=10_000,
    )
    gc.collect()

    # Run pilot for S3
    run_pilot_for_source(
        "s3", "S3",
        blocking_dir / "s1_s3_candidates.parquet",
        s1_lookup, s3_lookup,
        gt_pairs_s3, gt_dict,
        n_s1_entities=10_000,
    )


if __name__ == "__main__":
    main()