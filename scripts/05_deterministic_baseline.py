"""
scripts/05_deterministic_baseline.py
=====================================
Business Entity Resolution - Task 5: Deterministic Baseline Evaluation

Evaluates exact-match deterministic rules using representations from Task 4.
Measures macro Precision, Recall, F0.5 per S1 entity, ambiguity/collisions,
and singleton false-positive rates on S1->S2, S1->S3, and Combined S1->(S2+S3).

Zero hardcoded machine/user paths. Portable across all environments.
"""

import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Reconfigure stdout to UTF-8 and line-buffering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Determine repository root and add src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.matching import (
    build_target_index,
    predict_for_s1,
    evaluate_predictions,
)
from business_entity_resolution.preprocessing.normalization import load_normalized_or_compute


# ============================================================
# 1. Dataset Path Discovery
# ============================================================

def find_dataset_dir(repo_root: Path) -> Path:
    search_paths: List[Path] = []
    for env_var in ("BER_DATA_DIR", "DATA_DIR"):
        env_val = os.environ.get(env_var)
        if env_val:
            search_paths.append(Path(env_val))

    search_paths.extend([
        repo_root / "data" / "student_resource" / "dataset" / "train",
        repo_root / "data" / "dataset" / "train",
        repo_root / "data" / "train",
        repo_root / "dataset" / "train",
    ])

    required_files = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv",
    ]

    for candidate in search_paths:
        if candidate.is_dir():
            missing = [f for f in required_files if not (candidate / f).is_file()]
            if not missing:
                return candidate.resolve()

    error_msg = [
        "FATAL: Required training dataset files not found!",
        f"Searched candidate locations: {[str(p) for p in search_paths]}",
    ]
    raise FileNotFoundError("\n".join(error_msg))


TRAIN_DIR = find_dataset_dir(REPO_ROOT)
OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Diagnostic Collision Analyzer
# ============================================================

def get_top_ambiguous_keys(
    df: pd.DataFrame, key_cols: List[str], top_n: int = 10
) -> pd.DataFrame:
    """
    Identifies the keys with the highest number of collision IDs in the target source.
    """
    valid_mask = pd.Series(True, index=df.index)
    for c in key_cols:
        valid_mask &= (df[c].notna()) & (df[c] != "")

    valid_df = df[valid_mask]
    if len(key_cols) == 1:
        col = key_cols[0]
        counts = valid_df.groupby(col)["entity_id"].count().reset_index()
        counts.columns = ["Key", "Record_Count"]
    else:
        counts = valid_df.groupby(key_cols)["entity_id"].count().reset_index()
        counts["Key"] = counts[key_cols].apply(lambda r: " + ".join(str(x) for x in r), axis=1)
        counts = counts.rename(columns={"entity_id": "Record_Count"})[["Key", "Record_Count"]]

    return counts.sort_values(by="Record_Count", ascending=False).head(top_n)


# ============================================================
# 5. Main Execution Pipeline
# ============================================================

def main():
    print("=" * 80)
    print("TASK 5: DETERMINISTIC BASELINE EVALUATION")
    print("=" * 80)
    print(f"Repository Root : {REPO_ROOT}")
    print(f"Dataset Dir     : {TRAIN_DIR}")
    print(f"Output Dir      : {OUTPUT_DIR}\n")

    # 1. Load Data
    s1, s2, s3 = load_normalized_or_compute(TRAIN_DIR, REPO_ROOT)
    gt = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        usecols=["source1_entity_id", "matched_entity_ids"],
        dtype=str,
    )

    print(f"Loaded Source 1       : {len(s1):,} records")
    print(f"Loaded Source 2       : {len(s2):,} records")
    print(f"Loaded Source 3       : {len(s3):,} records")
    print(f"Loaded Ground Truth   : {len(gt):,} reference rows\n")

    # 3. Parse Ground Truth mappings
    print("\nParsing ground truth into entity sets...")
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

    truth_s2: Dict[str, Set[str]] = {}
    truth_s3: Dict[str, Set[str]] = {}
    truth_all: Dict[str, Set[str]] = {}

    s1_ids = s1["entity_id"].tolist()
    # Initialize all S1 entities with empty sets
    for eid in s1_ids:
        truth_s2[eid] = set()
        truth_s3[eid] = set()
        truth_all[eid] = set()

    for row in gt.itertuples(index=False):
        s1_id = getattr(row, "source1_entity_id")
        matched_str = getattr(row, "matched_entity_ids")
        if not s1_id or not matched_str or pd.isna(matched_str) or matched_str.strip() == "":
            continue
        m_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
        for m in m_ids:
            truth_all.setdefault(s1_id, set()).add(m)
            if m.startswith("S2-"):
                truth_s2.setdefault(s1_id, set()).add(m)
            elif m.startswith("S3-"):
                truth_s3.setdefault(s1_id, set()).add(m)

    n_s1 = len(s1_ids)
    s1_with_s2 = sum(1 for eid in s1_ids if len(truth_s2[eid]) > 0)
    s1_with_s3 = sum(1 for eid in s1_ids if len(truth_s3[eid]) > 0)
    s1_with_any = sum(1 for eid in s1_ids if len(truth_all[eid]) > 0)
    s1_singletons = n_s1 - s1_with_any

    print(f"Total S1 Entities           : {n_s1:,}")
    print(f"S1 with true S2 match(es)   : {s1_with_s2:,} ({s1_with_s2 / n_s1 * 100:.2f}%)")
    print(f"S1 with true S3 match(es)   : {s1_with_s3:,} ({s1_with_s3 / n_s1 * 100:.2f}%)")
    print(f"S1 with ANY true match      : {s1_with_any:,} ({s1_with_any / n_s1 * 100:.2f}%)")
    print(f"S1 Singletons (0 true match): {s1_singletons:,} ({s1_singletons / n_s1 * 100:.2f}%)\n")

    # 4. Define Rules to Evaluate
    rules: List[Tuple[str, List[str]]] = [
        ("A. name_normalized", ["name_norm"]),
        ("B. name_clean_legal", ["name_clean_legal"]),
        ("C. address_normalized", ["address_norm"]),
        ("D. name_norm + country_norm", ["name_norm", "country_norm"]),
        ("E. name_clean_legal + country_norm", ["name_clean_legal", "country_norm"]),
        ("F. name_norm + address_norm", ["name_norm", "address_norm"]),
        ("G. name_clean_legal + address_norm", ["name_clean_legal", "address_norm"]),
        ("H. name_clean_legal + address_norm + country_norm", ["name_clean_legal", "address_norm", "country_norm"]),
    ]

    results: List[Dict[str, Any]] = []
    ambiguous_diagnostics: List[pd.DataFrame] = []

    # 5. Evaluate Rules
    print("=" * 80)
    print("EVALUATING DETERMINISTIC RULES")
    print("=" * 80)

    for rule_name, key_cols in rules:
        print(f"\n---> Evaluating Rule: {rule_name} (keys: {key_cols})")

        # Build indices for S2 and S3
        idx_s2 = build_target_index(s2, key_cols)
        idx_s3 = build_target_index(s3, key_cols)

        # Predict
        preds_s2 = predict_for_s1(s1, key_cols, idx_s2)
        preds_s3 = predict_for_s1(s1, key_cols, idx_s3)

        # Combined predictions: union of S2 and S3 predicted IDs
        preds_all = [p2 + p3 for p2, p3 in zip(preds_s2, preds_s3)]

        # Evaluate S2
        res_s2 = evaluate_predictions(s1_ids, preds_s2, truth_s2, rule_name, "Source 2")
        results.append(res_s2)

        # Evaluate S3
        res_s3 = evaluate_predictions(s1_ids, preds_s3, truth_s3, rule_name, "Source 3")
        results.append(res_s3)

        # Evaluate Combined
        res_all = evaluate_predictions(s1_ids, preds_all, truth_all, rule_name, "Combined (S2+S3)")
        results.append(res_all)

        print(
            f"  [S2]  Coverage: {res_s2['Coverage_Pct']:5.2f}% | "
            f"Ambiguous: {res_s2['Ambiguous_Pct_Total']:5.2f}% | "
            f"Max Cand: {res_s2['Max_Candidates']:>4} | "
            f"Prec: {res_s2['Macro_Precision']:.4f} | "
            f"Rec: {res_s2['Macro_Recall']:.4f} | "
            f"F0.5: {res_s2['Macro_F0.5']:.4f} | "
            f"Sing. FP Rate: {res_s2['Singleton_FP_Rate'] * 100:.2f}%"
        )
        print(
            f"  [S3]  Coverage: {res_s3['Coverage_Pct']:5.2f}% | "
            f"Ambiguous: {res_s3['Ambiguous_Pct_Total']:5.2f}% | "
            f"Max Cand: {res_s3['Max_Candidates']:>4} | "
            f"Prec: {res_s3['Macro_Precision']:.4f} | "
            f"Rec: {res_s3['Macro_Recall']:.4f} | "
            f"F0.5: {res_s3['Macro_F0.5']:.4f} | "
            f"Sing. FP Rate: {res_s3['Singleton_FP_Rate'] * 100:.2f}%"
        )
        print(
            f"  [ALL] Coverage: {res_all['Coverage_Pct']:5.2f}% | "
            f"Ambiguous: {res_all['Ambiguous_Pct_Total']:5.2f}% | "
            f"Max Cand: {res_all['Max_Candidates']:>4} | "
            f"Prec: {res_all['Macro_Precision']:.4f} | "
            f"Rec: {res_all['Macro_Recall']:.4f} | "
            f"F0.5: {res_all['Macro_F0.5']:.4f} | "
            f"Sing. FP Rate: {res_all['Singleton_FP_Rate'] * 100:.2f}%"
        )

        # Collect top collisions for diagnostics
        top_s2 = get_top_ambiguous_keys(s2, key_cols, top_n=5)
        top_s2["Source"] = "Source 2"
        top_s2["Rule"] = rule_name
        ambiguous_diagnostics.append(top_s2)

    # 6. Save Results
    results_df = pd.DataFrame(results)
    results_csv = OUTPUT_DIR / "deterministic_baseline_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSuccessfully saved detailed benchmark results to:\n  {results_csv}")

    diag_df = pd.concat(ambiguous_diagnostics, ignore_index=True)
    diag_csv = OUTPUT_DIR / "ambiguous_keys_diagnostic.csv"
    diag_df.to_csv(diag_csv, index=False)
    print(f"Saved diagnostic collision table to:\n  {diag_csv}\n")

    # 7. Summary Tables
    print("=" * 110)
    print("SUMMARY BENCHMARK TABLE: COMBINED (S2 + S3)")
    print("=" * 110)
    combined_df = results_df[results_df["Source"] == "Combined (S2+S3)"].copy()
    display_cols = [
        "Rule",
        "Coverage_Pct",
        "Ambiguous_Pct_Total",
        "Max_Candidates",
        "Mean_Candidates_Matched",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F0.5",
        "Singleton_FP_Rate",
        "Exact_Match_Count",
        "Completely_Correct_Multi",
    ]
    print(combined_df[display_cols].to_string(index=False))

    print("\n" + "=" * 110)
    print("SUMMARY BENCHMARK TABLE: SOURCE 2 vs SOURCE 3")
    print("=" * 110)
    s2_s3_df = results_df[results_df["Source"].isin(["Source 2", "Source 3"])].copy()
    s2_s3_display = [
        "Rule",
        "Source",
        "Coverage_Pct",
        "Ambiguous_Pct_Total",
        "Max_Candidates",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F0.5",
        "Singleton_FP_Rate",
    ]
    print(s2_s3_df[s2_s3_display].to_string(index=False))

    # 8. Interpretation & Key Takeaways
    print("\n" + "=" * 80)
    print("TASK 5 INTERPRETATION & KEY FINDINGS")
    print("=" * 80)

    # Find best F0.5 rule
    best_f05_row = combined_df.sort_values(by="Macro_F0.5", ascending=False).iloc[0]
    best_rec_row = combined_df.sort_values(by="Macro_Recall", ascending=False).iloc[0]
    best_prec_row = combined_df.sort_values(by="Macro_Precision", ascending=False).iloc[0]

    print(f"1. HIGHEST RECALL RULE:")
    print(f"   -> '{best_rec_row['Rule']}' with Macro Recall = {best_rec_row['Macro_Recall']:.4f} (Pos Recall: {best_rec_row['Pos_Macro_Recall']:.4f}, Coverage: {best_rec_row['Coverage_Pct']:.2f}%).")
    print(f"   Tradeoff: Stripping legal suffixes increases recall, but can introduce name ambiguity if not anchored with country/address.\n")

    print(f"2. HIGHEST PRECISION / F0.5 RULE:")
    print(f"   -> Best Macro F0.5: '{best_f05_row['Rule']}' (Macro F0.5 = {best_f05_row['Macro_F0.5']:.4f}, Macro Precision = {best_f05_row['Macro_Precision']:.4f}).")
    print(f"   -> Best Precision: '{best_prec_row['Rule']}' (Macro Precision = {best_prec_row['Macro_Precision']:.4f}, Singleton FP Rate = {best_prec_row['Singleton_FP_Rate'] * 100:.2f}%).\n")

    print(f"3. DANGEROUS AMBIGUITY RULES:")
    high_ambig = combined_df.sort_values(by="Ambiguous_Pct_Total", ascending=False).iloc[0]
    print(f"   -> '{high_ambig['Rule']}' produced {high_ambig['Ambiguous_Count']:,} ambiguous S1 entities with max candidate count {high_ambig['Max_Candidates']:,}!")
    print(f"   Address-only matching (Rule C) and single-token name collisions create massive false-positive explosions.\n")

    print(f"4. UNRESOLVED PROBLEM SIZE:")
    unresolved_entities = n_s1 - best_f05_row["Exact_Match_Count"]
    print(f"   -> Exact matching alone leaves {unresolved_entities:,} / {n_s1:,} ({unresolved_entities / n_s1 * 100:.2f}%) S1 entities unresolved or imperfect.")
    print(f"   -> Even the best exact rule recovers only a fraction of positive true pairs because real-world entity noise includes spelling typos, address permutations, and missing fields.\n")

    print(f"5. IMPLICATIONS FOR BLOCKING & ML:")
    print(f"   - Strict exact matching is high precision but suffers from severe recall ceilings.")
    print(f"   - Loose exact keys (like clean name alone or address alone) cause catastrophic collision bursts.")
    print(f"   - Next stage (Candidate Generation / Blocking) must cast a multi-key net (e.g. phonetic, token/n-gram, geo/country blocks) while ML / scoring handles disambiguation with high precision.")
    print("=" * 80)


if __name__ == "__main__":
    main()
