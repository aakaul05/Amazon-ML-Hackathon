"""
scripts/run_v3_lightgbm_pipeline.py
===================================
Master Orchestrator for the Complete V3 Blocking + LightGBM Matching & Submission Pipeline.

Target: Push Leaderboard Macro F0.5 toward 0.92+ by combining:
  1. High-Recall Lean V3 Blocking (>92% recall, compound address, compact prefix, token pairs)
  2. 5-Fold Entity-Level LightGBM Ensemble (calibrated at threshold 0.70)
  3. Strict Deliverable Verification & Official Validator Check

Pipeline Steps:
  Step 1: Check Pre-trained 5-Fold LightGBM Ensemble
          - Verified in data/student_resource/outputs/matching/models/lightgbm/
  Step 2: Test Data Normalization
          - python scripts/09_test_normalization.py (skips if cached)
  Step 3: Test Candidate Generation via V3 Blocking
          - python scripts/10_test_blocking_v3.py
  Step 4: Streaming 30-Feature Inference & LightGBM Ensemble
          - python scripts/11_test_inference_lightgbm.py --threshold 0.70
  Step 5: Create Official Deliverables & Run Official Validator
          - python scripts/12_create_submission.py

Usage:
  python3 scripts/run_v3_lightgbm_pipeline.py
  python3 scripts/run_v3_lightgbm_pipeline.py --start-from-step 3
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
        "name": "Verify LightGBM Ensemble Models",
        "description": "Checks existing 5 LightGBM fold models and metrics",
    },
    {
        "step_num": 2,
        "name": "Test Data Normalization",
        "script": REPO_ROOT / "scripts" / "09_test_normalization.py",
        "description": "Normalizes test_source1..3.tsv (resumes/skips if cached)",
    },
    {
        "step_num": 3,
        "name": "Test Candidate Generation (V3 High-Recall Blocking)",
        "script": REPO_ROOT / "scripts" / "10_test_blocking_v3.py",
        "description": "Generates test candidate pairs via V3 blocking with >92% recall",
    },
    {
        "step_num": 4,
        "name": "Streaming 30-Feature Inference & LightGBM Ensemble",
        "script": REPO_ROOT / "scripts" / "11_test_inference_lightgbm.py",
        "description": "500k-chunk streaming inference with 5 LightGBM models at optimal threshold",
    },
    {
        "step_num": 5,
        "name": "Create Official Deliverables & Validate",
        "script": REPO_ROOT / "scripts" / "12_create_submission.py",
        "description": "Generates output/matching_results.tsv & candidate_pairs.tsv and executes validator",
    },
]


def check_models_exist() -> bool:
    models_dir = REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models" / "lightgbm"
    if not models_dir.is_dir():
        models_dir = REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models"
    for fold in range(1, 6):
        m = models_dir / f"lightgbm_matcher_fold{fold}.txt"
        if not m.is_file():
            return False
    return True


def run_step(step_info: dict) -> bool:
    step_num = step_info["step_num"]
    name = step_info["name"]
    desc = step_info["description"]

    print("\n" + "=" * 80)
    print(f"PIPELINE STEP {step_num}/5: {name.upper()}")
    print(f"Description: {desc}")
    print("=" * 80)

    if step_num == 1:
        if check_models_exist():
            print("  [OK] All 5 LightGBM models found and verified.")
            return True
        else:
            print("  [WARN] LightGBM models not found in default dir. Training may be required.")
            train_script = REPO_ROOT / "scripts" / "08c_train_lightgbm.py"
            cmd = [sys.executable, "-u", str(train_script)]
            res = subprocess.run(cmd)
            return res.returncode == 0

    script_path = step_info.get("script")
    if not script_path or not script_path.is_file():
        print(f"ERROR: Script not found: {script_path}")
        return False

    cmd = [sys.executable, "-u", str(script_path)]
    t0 = time.time()
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[FAIL] Step {step_num} failed with return code {result.returncode} after {elapsed:.1f}s")
        return False

    print(f"\n[SUCCESS] Step {step_num} completed successfully in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run V3 Blocking + LightGBM Test Pipeline")
    parser.add_argument("--start-from-step", type=int, default=1, choices=range(1, 6), help="Start from step (1-5)")
    parser.add_argument("--step", type=int, default=None, choices=range(1, 6), help="Run only specific step")
    args = parser.parse_args()

    t_all = time.time()
    print("=" * 80)
    print("V3 BLOCKING + LIGHTGBM TEST PIPELINE ORCHESTRATOR")
    print("TARGET SCORE: 0.92+ MACRO F0.5")
    print("=" * 80)

    steps_to_run = [s for s in PIPELINE_STEPS if args.step == s["step_num"]] if args.step else [s for s in PIPELINE_STEPS if s["step_num"] >= args.start_from_step]

    for step in steps_to_run:
        success = run_step(step)
        if not success:
            print(f"\nPipeline halted at Step {step['step_num']}.")
            sys.exit(1)

    print("\n" + "=" * 80)
    print(f"ALL PIPELINE STEPS COMPLETED IN {time.time() - t_all:.1f}s ({(time.time() - t_all) / 60:.1f} min)")
    print("Official submission files generated in: output/")
    print("  - output/matching_results.tsv")
    print("  - output/candidate_pairs.tsv")
    print("=" * 80)


if __name__ == "__main__":
    main()
