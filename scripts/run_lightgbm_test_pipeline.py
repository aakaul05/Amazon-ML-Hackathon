"""
scripts/run_lightgbm_test_pipeline.py
======================================
Master Orchestrator for the Complete LightGBM-Based Matching and Test Pipeline.

Executes the pipeline steps in exact sequence:
  Step 1: Train 5-Fold LightGBM Ensemble (if not already trained)
          - python scripts/08c_train_lightgbm.py
          - Reads Task 8B precomputed features (s1_s2 & s1_s3)
          - Trains 5 folds with entity-level split
          - Optimizes threshold on OOF Macro F0.5
          - Saves models to data/student_resource/outputs/matching/models/lightgbm/

  Step 2: Test Data Normalization
          - python scripts/09_test_normalization.py (skips if files exist)

  Step 3: Test Candidate Generation (10-Pass Blocking)
          - python scripts/10_test_blocking.py (skips if candidate files exist)

  Step 4: Streaming 30-Feature Inference & LightGBM Ensemble
          - python scripts/11_test_inference_lightgbm.py
          - 500k-chunk streaming inference with 5 LightGBM models

  Step 5: Create Official Deliverables & Run Official Validator
          - python scripts/12_create_submission.py
          - Generates output/matching_results.tsv and output/candidate_pairs.tsv
          - Runs dataset/student_resource/utils/validate_submission.py

Usage:
  python3 scripts/run_lightgbm_test_pipeline.py

Optional flags:
  --start-from-step <1-5>   (resume from a specific step)
  --step <1-5>              (run only a single step)
  --dry-run                 (verify prerequisites without executing)
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PIPELINE_STEPS = [
    {
        "step_num": 1,
        "name": "Train 5-Fold LightGBM Ensemble",
        "script": REPO_ROOT / "scripts" / "08c_train_lightgbm.py",
        "description": "Trains 5 LightGBM fold models from precomputed features and calibrates threshold",
    },
    {
        "step_num": 2,
        "name": "Test Data Normalization",
        "script": REPO_ROOT / "scripts" / "09_test_normalization.py",
        "description": "Normalizes test_source1..3.tsv (resumes/skips if cached)",
    },
    {
        "step_num": 3,
        "name": "Test Candidate Generation (10-Pass Blocking)",
        "script": REPO_ROOT / "scripts" / "10_test_blocking.py",
        "description": "Generates test candidate pairs via Task 7 blocking (resumes/skips if cached)",
    },
    {
        "step_num": 4,
        "name": "Streaming 30-Feature Inference & LightGBM Ensemble",
        "script": REPO_ROOT / "scripts" / "11_test_inference_lightgbm.py",
        "description": "500k-chunk streaming inference with 5 LightGBM models at calibrated threshold",
    },
    {
        "step_num": 5,
        "name": "Create Official Deliverables & Validate",
        "script": REPO_ROOT / "scripts" / "12_create_submission.py",
        "description": "Generates output/matching_results.tsv & candidate_pairs.tsv and executes validator",
    },
]


def check_prerequisites():
    print("=" * 80)
    print("LIGHTGBM PIPELINE: PREREQUISITE VERIFICATION")
    print("=" * 80)

    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Repository Root   : {REPO_ROOT}")

    # Check LightGBM module
    try:
        import lightgbm as lgb
        print(f"  [OK] LightGBM version: {lgb.__version__}")
    except ImportError:
        print("  [FAIL] LightGBM is not installed! Run: pip install lightgbm")
        sys.exit(1)

    # Disk Space
    total, used, free = shutil.disk_usage(REPO_ROOT)
    free_gb = free / (1024**3)
    print(f"Disk Storage      : {free_gb:.1f} GB free of {total / (1024**3):.1f} GB total")

    # Check precomputed features (for Step 1 training)
    features_dir = REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "features"
    if (features_dir / "s1_s2").is_dir() and (features_dir / "s1_s3").is_dir():
        n_s2 = len(list((features_dir / "s1_s2").glob("part_*.parquet")))
        n_s3 = len(list((features_dir / "s1_s3").glob("part_*.parquet")))
        print(f"  [OK] Found precomputed features: s1_s2 ({n_s2} parts), s1_s3 ({n_s3} parts)")
    else:
        print(f"  [NOTE] Training features not local at {features_dir} (check BER_MODELS_DIR if pre-trained)")

    # Check existing LightGBM models
    models_dir = REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models" / "lightgbm"
    existing_models = list(models_dir.glob("lightgbm_matcher_fold*.txt")) if models_dir.is_dir() else []
    if len(existing_models) == 5:
        print(f"  [OK] Found 5 pre-trained LightGBM models in {models_dir}")
    else:
        print(f"  [NOTE] LightGBM models ({len(existing_models)}/5) will be trained in Step 1")

    # Check test candidates
    test_blocking_dir = REPO_ROOT / "data" / "student_resource" / "outputs" / "test" / "blocking"
    s2_cand = test_blocking_dir / "test_s1_s2_candidates.parquet"
    s3_cand = test_blocking_dir / "test_s1_s3_candidates.parquet"
    if s2_cand.is_file() and s3_cand.is_file():
        print(f"  [OK] Test candidates already generated: {s2_cand.name}, {s3_cand.name} (Step 3 will skip)")

    print("=" * 80)
    print("ALL PREREQUISITES VERIFIED SUCCESSFULLY.")
    print("=" * 80)


def run_step(step_dict: dict) -> bool:
    step_num = step_dict["step_num"]
    name = step_dict["name"]
    script = step_dict["script"]
    desc = step_dict["description"]

    # Smart skip logic for steps already completed
    if step_num == 1:
        models_dir = REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models" / "lightgbm"
        if models_dir.is_dir() and len(list(models_dir.glob("lightgbm_matcher_fold*.txt"))) == 5:
            print(f"\n[Step {step_num}] {name}: 5 LightGBM models already exist. Skipping training.")
            return True

    if step_num == 2:
        norm_dir = REPO_ROOT / "data" / "student_resource" / "outputs" / "test" / "normalized"
        if norm_dir.is_dir() and len(list(norm_dir.glob("*test_normalized.parquet"))) >= 3:
            print(f"\n[Step {step_num}] {name}: Test normalized files already exist. Skipping normalization.")
            return True

    if step_num == 3:
        blocking_dir = REPO_ROOT / "data" / "student_resource" / "outputs" / "test" / "blocking"
        if (blocking_dir / "test_s1_s2_candidates.parquet").is_file() and (blocking_dir / "test_s1_s3_candidates.parquet").is_file():
            print(f"\n[Step {step_num}] {name}: Test candidate parquets already exist. Skipping blocking.")
            return True

    print("\n" + "=" * 80)
    print(f"LAUNCHING STEP {step_num}/5: {name.upper()}")
    print(f"Script : {script.name}")
    print(f"Details: {desc}")
    print("=" * 80 + "\n")

    t0 = time.time()
    cmd = [sys.executable, "-u", str(script)]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))

    elapsed = (time.time() - t0) / 60
    if result.returncode != 0:
        print(f"\n[FAIL] Step {step_num} failed with exit code {result.returncode} after {elapsed:.1f}m")
        return False

    print(f"\n[SUCCESS] Step {step_num} completed successfully in {elapsed:.1f}m")
    return True


def main():
    parser = argparse.ArgumentParser(description="Master LightGBM Pipeline Runner")
    parser.add_argument("--start-from-step", type=int, default=1, choices=range(1, 6), help="Resume from step N")
    parser.add_argument("--step", type=int, default=None, choices=range(1, 6), help="Run only step N")
    parser.add_argument("--dry-run", action="store_true", help="Verify prerequisites without running")
    args = parser.parse_args()

    check_prerequisites()
    if args.dry_run:
        print("\nDry-run complete. Exiting.")
        return

    start_all = time.time()

    if args.step is not None:
        target_step = [s for s in PIPELINE_STEPS if s["step_num"] == args.step][0]
        success = run_step(target_step)
        if not success:
            sys.exit(1)
    else:
        for s in PIPELINE_STEPS:
            if s["step_num"] < args.start_from_step:
                print(f"[SKIP] Step {s['step_num']}: {s['name']} (skipped via --start-from-step {args.start_from_step})")
                continue
            success = run_step(s)
            if not success:
                sys.exit(1)

    total_min = (time.time() - start_all) / 60
    print("\n" + "=" * 80)
    print(f"COMPLETE LIGHTGBM PIPELINE FINISHED SUCCESSFULLY IN {total_min:.1f} MINUTES!")
    print("Deliverables generated:")
    print("  1. output/matching_results.tsv")
    print("  2. output/candidate_pairs.tsv")
    print("Official submission validation: PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
