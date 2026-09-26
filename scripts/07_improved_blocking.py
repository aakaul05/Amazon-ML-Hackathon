"""
scripts/07_improved_blocking.py
================================
Business Entity Resolution — Task 7: Improved Blocking (10 Passes)

Goal: Push blocking recall from ~57% → 80-90%+ by adding 5 new passes
to the original 5. With 32GB RAM, we load everything at once.

10 Blocking Passes:
 1. Exact name_norm
 2. Exact name_clean_legal
 3. Rare/Informative Name Tokens
 4. Address Component Tokens
 5. TF-IDF char n-gram (threshold 0.70, top_k=10)
 6. TF-IDF char n-gram (relaxed 0.45, top_k=20) — fuzzy misses
 7. Sorted-Token Blocking — word reordering
 8. Name Prefix (first 6 chars) — abbreviations
 9. Country + Name-token compound — geographic anchoring
10. Phonetic Blocking — transliterations / spelling

Outputs (replaces Task 6 outputs):
 - data/student_resource/outputs/blocking/s1_s2_candidates.parquet
 - data/student_resource/outputs/blocking/s1_s3_candidates.parquet
 - data/student_resource/outputs/blocking/blocking_statistics_v2.csv
"""

import gc
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd

# Reconfigure stdout for UTF-8 line buffering on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.blocking import (
    block_exact_field,
    block_rare_tokens,
    block_address_tokens,
    block_tfidf_char_ngram,
    block_sorted_tokens,
    block_name_prefix,
    block_country_name_token,
    block_phonetic,
    combine_blocks_and_evaluate,
)
from business_entity_resolution.preprocessing.normalization import (
    load_normalized_or_compute,
)

OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "blocking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Path Discovery
# ============================================================

def find_dataset_dir():
    for env_var in ("BER_DATA_DIR", "DATA_DIR"):
        env_val = os.environ.get(env_var)
        if env_val:
            p = Path(env_val)
            if p.is_dir():
                return p.resolve()

    candidates = [
        REPO_ROOT / "data" / "student_resource" / "dataset" / "train",
        REPO_ROOT / "data" / "dataset" / "train",
        REPO_ROOT / "data" / "train",
        REPO_ROOT / "dataset" / "student_resource" / "dataset" / "train",
        REPO_ROOT / "dataset" / "train",
    ]

    for p in candidates:
        if p.is_dir() and (p / "train_ground_truth.tsv").is_file():
            return p.resolve()

    raise FileNotFoundError("Training dataset directory not found. Set BER_DATA_DIR.")


TRAIN_DIR = find_dataset_dir()


# ============================================================
# Ground Truth Parsing
# ============================================================

def parse_ground_truth():
    """Stream-parse ground truth into S2 and S3 dicts."""
    gt_s2 = defaultdict(set)
    gt_s3 = defaultdict(set)
    total_s2 = 0
    total_s3 = 0

    path = TRAIN_DIR / "train_ground_truth.tsv"
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            s1_id = parts[0].strip()
            matches = parts[1].strip()
            if not s1_id or not matches:
                continue
            for m in matches.split(","):
                m = m.strip()
                if m.startswith("S2-"):
                    gt_s2[s1_id].add(m)
                    total_s2 += 1
                elif m.startswith("S3-"):
                    gt_s3[s1_id].add(m)
                    total_s3 += 1

    return gt_s2, gt_s3, total_s2, total_s3


# ============================================================
# Block Runner
# ============================================================

def run_pass(name, func):
    """Run a blocking pass with timing and error handling."""
    print("\n" + "=" * 70, flush=True)
    print(f"STARTING {name}", flush=True)
    print("=" * 70, flush=True)

    start = time.time()
    try:
        result = func()
        print(f"FINISHED {name} in {time.time() - start:.1f}s", flush=True)
        gc.collect()
        return result
    except Exception as e:
        print(f"\nERROR IN {name}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Return empty DataFrame so the pipeline continues
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "block"])


# ============================================================
# Run All 10 Blocking Passes for a Source
# ============================================================

def run_all_blocks(s1, other, source_label):
    """Run all 10 blocking passes for S1 vs other source."""

    print(f"\n{'#' * 70}")
    print(f"# BLOCKING: S1 → {source_label}")
    print(f"# S1: {len(s1):,} rows  |  {source_label}: {len(other):,} rows")
    print(f"{'#' * 70}")

    blocks = []

    # ── Pass 1: Exact name_norm ──────────────────────────────
    blocks.append(run_pass(
        "Pass 1 — Exact name_norm",
        lambda: block_exact_field(s1, other, "name_norm", "exact_name_norm"),
    ))

    # ── Pass 2: Exact clean legal name ───────────────────────
    blocks.append(run_pass(
        "Pass 2 — Exact clean legal name",
        lambda: block_exact_field(s1, other, "name_clean_legal", "exact_clean_legal"),
    ))

    # ── Pass 3: Rare tokens (max_df=500) ─────────────────────
    blocks.append(run_pass(
        "Pass 3 — Rare tokens (DF<=500)",
        lambda: block_rare_tokens(s1, other, max_df=500, max_cand_per_s1=40),
    ))

    # ── Pass 4: Address tokens ───────────────────────────────
    blocks.append(run_pass(
        "Pass 4 — Address tokens",
        lambda: block_address_tokens(s1, other),
    ))

    # ── Pass 5: TF-IDF char n-gram (high threshold) ─────────
    blocks.append(run_pass(
        "Pass 5 — TF-IDF char 3-gram (sim>=0.70, top_k=10)",
        lambda: block_tfidf_char_ngram(
            s1, other, min_sim=0.70, top_k=10, sample_limit=300000,
        ),
    ))

    # ── Pass 6: TF-IDF char n-gram (RELAXED threshold) ──────
    blocks.append(run_pass(
        "Pass 6 — TF-IDF char 3-gram RELAXED (sim>=0.50, top_k=10)",
        lambda: block_tfidf_char_ngram(
            s1, other, min_sim=0.50, top_k=10, sample_limit=300000,
        ),
    ))

    # ── Pass 7: Sorted-token blocking ────────────────────────
    blocks.append(run_pass(
        "Pass 7 — Sorted-token blocking",
        lambda: block_sorted_tokens(s1, other, max_key_df=300, max_cand_per_s1=40),
    ))

    # ── Pass 8: Name prefix blocking (6 chars) ───────────────
    blocks.append(run_pass(
        "Pass 8 — Name prefix (6 chars)",
        lambda: block_name_prefix(s1, other, prefix_len=6, max_key_df=150, max_cand_per_s1=25),
    ))

    # ── Pass 9: Country + name token compound ────────────────
    blocks.append(run_pass(
        "Pass 9 — Country + name token compound",
        lambda: block_country_name_token(
            s1, other, max_key_df=250, max_cand_per_s1=30, min_token_len=4,
        ),
    ))

    # ── Pass 10: Phonetic blocking ───────────────────────────
    blocks.append(run_pass(
        "Pass 10 — Phonetic blocking",
        lambda: block_phonetic(s1, other, max_key_df=200, max_cand_per_s1=30),
    ))

    return blocks


# ============================================================
# Main Pipeline
# ============================================================

def main():
    t_total = time.time()

    print("=" * 70)
    print("TASK 7: IMPROVED BLOCKING (10 PASSES)")
    print("=" * 70)
    print(f"Dataset : {TRAIN_DIR}")
    print(f"Output  : {OUTPUT_DIR}")

    # ── Ground Truth ──────────────────────────────────────────
    gt_s2, gt_s3, total_s2, total_s3 = parse_ground_truth()
    print(f"\nGround truth S2 pairs: {total_s2:,}")
    print(f"Ground truth S3 pairs: {total_s3:,}")

    # ── Load ALL sources (32GB RAM available) ─────────────────
    cols = ["entity_id", "name_norm", "name_clean_legal", "address_norm", "country_norm"]
    s1, s2, s3 = load_normalized_or_compute(TRAIN_DIR, REPO_ROOT, columns=cols)

    print(f"\nS1 rows: {len(s1):,}")
    print(f"S2 rows: {len(s2):,}")
    print(f"S3 rows: {len(s3):,}")

    # ── SOURCE 2 BLOCKING ────────────────────────────────────
    blocks_s2 = run_all_blocks(s1, s2, "Source 2")

    print("\n" + "=" * 70)
    print("EVALUATING SOURCE 2 UNION")
    print("=" * 70)

    cand_s2, stats_s2 = combine_blocks_and_evaluate(
        s1, s2, gt_s2, total_s2, blocks_s2, "Source 2",
    )

    out_s2 = OUTPUT_DIR / "s1_s2_candidates.parquet"
    cand_s2.to_parquet(out_s2, index=False)
    print(f"\nSaved: {out_s2}  ({len(cand_s2):,} pairs)")

    # Free S2 resources
    del s2, blocks_s2, cand_s2
    gc.collect()

    # ── SOURCE 3 BLOCKING ────────────────────────────────────
    blocks_s3 = run_all_blocks(s1, s3, "Source 3")

    print("\n" + "=" * 70)
    print("EVALUATING SOURCE 3 UNION")
    print("=" * 70)

    cand_s3, stats_s3 = combine_blocks_and_evaluate(
        s1, s3, gt_s3, total_s3, blocks_s3, "Source 3",
    )

    out_s3 = OUTPUT_DIR / "s1_s3_candidates.parquet"
    cand_s3.to_parquet(out_s3, index=False)
    print(f"\nSaved: {out_s3}  ({len(cand_s3):,} pairs)")

    del s1, s3, blocks_s3, cand_s3
    gc.collect()

    # ── Save Statistics ──────────────────────────────────────
    stats_df = pd.DataFrame([stats_s2, stats_s3])
    stats_out = OUTPUT_DIR / "blocking_statistics_v2.csv"
    stats_df.to_csv(stats_out, index=False)
    print(f"\nSaved: {stats_out}")

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"TASK 7 COMPLETE — Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")

    print(f"\n{'SOURCE':<12} {'RECALL':>8} {'TP':>12} {'TOTAL_GT':>12} {'CANDIDATES':>14} {'COMPLETE%':>10} {'ZERO%':>8}")
    print("-" * 80)
    for s in [stats_s2, stats_s3]:
        print(
            f"{s['source']:<12} "
            f"{s['candidate_recall']*100:>7.2f}% "
            f"{s['total_tp_recovered']:>12,} "
            f"{s['total_true_pairs']:>12,} "
            f"{s['total_candidates']:>14,} "
            f"{s['complete_s1_coverage_pct']:>9.2f}% "
            f"{s['zero_s1_coverage_pct']:>7.2f}%"
        )


if __name__ == "__main__":
    main()
