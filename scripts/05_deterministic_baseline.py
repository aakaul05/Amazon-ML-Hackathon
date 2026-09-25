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

from business_entity_resolution.preprocessing.normalization import normalize_source_df


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
# 2. Fast Target Index Builder & Prediction
# ============================================================

def build_target_index(df: pd.DataFrame, key_cols: List[str]) -> Dict[Any, List[str]]:
    """
    Constructs a fast lookup dictionary: key -> list of candidate entity_ids.
    Guards against empty strings: rows with any empty/NaN key component are excluded.
    """
    # Create mask where all key cols are non-empty
    valid_mask = pd.Series(True, index=df.index)
    for c in key_cols:
        valid_mask &= (df[c].notna()) & (df[c] != "")

    valid_df = df[valid_mask]
    if len(valid_df) == 0:
        return {}

    if len(key_cols) == 1:
        keys = valid_df[key_cols[0]].values
    else:
        keys = list(zip(*(valid_df[c].values for c in key_cols)))
    ids = valid_df["entity_id"].values

    index: Dict[Any, List[str]] = {}
    for k, eid in zip(keys, ids):
        if k in index:
            index[k].append(eid)
        else:
            index[k] = [eid]

    return index


def predict_for_s1(
    s1_df: pd.DataFrame,
    key_cols: List[str],
    target_index: Dict[Any, List[str]],
) -> List[List[str]]:
    """
    Retrieves all matching IDs for each S1 entity given a deterministic rule index.
    Guards against empty keys in S1 (returns [] if any key column is empty).
    """
    valid_mask = pd.Series(True, index=s1_df.index)
    for c in key_cols:
        valid_mask &= (s1_df[c].notna()) & (s1_df[c] != "")

    if len(key_cols) == 1:
        s1_keys = s1_df[key_cols[0]].values
    else:
        s1_keys = list(zip(*(s1_df[c].values for c in key_cols)))

    preds: List[List[str]] = []
    valid_vals = valid_mask.values
    for is_valid, k in zip(valid_vals, s1_keys):
        if is_valid and k in target_index:
            preds.append(target_index[k])
        else:
            preds.append([])

    return preds


# ============================================================
# 3. Entity-Level Evaluation Engine
# ============================================================

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
    coverage_pct = (s1_matched_count / n_entities) * 100.0

    ambiguous_mask = pred_counts > 1
    s1_ambiguous_count = int(np.sum(ambiguous_mask))
    ambiguous_pct_total = (s1_ambiguous_count / n_entities) * 100.0
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

    # Positive entity metrics (for diagnostic breakdown)
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
                # Correct singleton
                precisions[i] = 1.0
                recalls[i] = 1.0
                f05s[i] = 1.0
                exact_match_count += 1
            else:
                # Singleton False Positive
                singleton_fp_count += 1
                precisions[i] = 0.0
                recalls[i] = 0.0
                f05s[i] = 0.0
        else:
            # Non-singleton entity (has positive matches in truth)
            if t_len > 1:
                multi_match_truth_count += 1

            if p_len == 0:
                # False negative (no prediction)
                precisions[i] = 0.0
                recalls[i] = 0.0
                f05s[i] = 0.0
                pos_precisions.append(0.0)
                pos_recalls.append(0.0)
                pos_f05s.append(0.0)
            else:
                tp = len(pred_set & true_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)

                p = tp / p_len
                r = tp / t_len
                precisions[i] = p
                recalls[i] = r

                if tp == 0:
                    f05 = 0.0
                else:
                    # Beta = 0.5 -> beta^2 = 0.25 -> (1 + 0.25) * P * R / (0.25 * P + R)
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

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f05 = float(np.mean(f05s))

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


# ============================================================
# 4. Diagnostic Collision Analyzer
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
    print("Loading datasets...")
    s1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype=str,
    )
    s2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype=str,
    )
    s3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype=str,
    )
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

    # 2. Normalize
    print("Normalizing Source 1...")
    s1 = normalize_source_df(s1)
    print("Normalizing Source 2...")
    s2 = normalize_source_df(s2)
    print("Normalizing Source 3...")
    s3 = normalize_source_df(s3)

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
