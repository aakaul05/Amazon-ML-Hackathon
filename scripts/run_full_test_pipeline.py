"""
scripts/run_full_test_pipeline.py
=================================
Master Orchestrator for the Complete Test/Inference Pipeline.

Executes the 4 verified production pipeline steps in exact sequence:
  Step 1: python scripts/09_test_normalization.py
          - Normalizes test_source1..3.tsv -> data/.../test/normalized/
  Step 2: python scripts/10_test_blocking.py
          - Runs 10 Task-7 blocking passes -> data/.../test/blocking/
  Step 3: python scripts/11_test_inference.py
          - Streaming 30-feature computation + 5-fold CatBoost ensemble inference (optimal threshold 0.95)
  Step 4: python scripts/12_create_submission.py
          - Aggregates deliverables to output/matching_results.tsv & output/candidate_pairs.tsv
          - Executes official submission validator

Usage on EC2:
  python3 scripts/run_full_test_pipeline.py

Optional flags:
  --start-from-step <1-4>   (resume from a specific step)
  --step <1-4>              (run only a single step)
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
        "name": "Test Data Normalization",
        "script": REPO_ROOT / "scripts" / "09_test_normalization.py",
        "description": "Normalizes test_source1.tsv, test_source2.tsv, test_source3.tsv",
    },
    {
        "step_num": 2,
        "name": "Test Candidate Generation (10-Pass Blocking)",
        "script": REPO_ROOT / "scripts" / "10_test_blocking.py",
        "description": "Generates candidate pairs using Task 7 blocking without ground truth",
    },
    {
        "step_num": 3,
        "name": "Streaming 30-Feature Inference & CatBoost Ensemble",
        "script": REPO_ROOT / "scripts" / "11_test_inference.py",
        "description": "500k-chunk streaming inference with 5 CatBoost models at threshold 0.98",
    },
    {
        "step_num": 4,
        "name": "Create Official Deliverables & Validation",
        "script": REPO_ROOT / "scripts" / "12_create_submission.py",
        "description": "Writes output/matching_results.tsv and candidate_pairs.tsv and runs official validator",
    },
]


def check_prerequisites():
    """Verifies all input files, models, and system resources before launching."""
    print("=" * 80)
    print("PREREQUISITE VERIFICATION")
    print("=" * 80)

    # 1. Python environment
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Repository Root   : {REPO_ROOT}")

    # 2. Disk Space
    total, used, free = shutil.disk_usage(REPO_ROOT)
    total_gb = total / (1024**3)
    free_gb = free / (1024**3)
    print(f"Disk Storage      : {free_gb:.1f} GB free of {total_gb:.1f} GB total")
    if free_gb < 10:
        print("  WARNING: Less than 10 GB free disk space available! Ensure sufficient storage.")
    else:
        print("  [OK] Sufficient disk space available.")

    # 3. Test Inputs
    candidates_test_dirs = [
        REPO_ROOT / "dataset" / "student_resource" / "dataset" / "test",
        Path("/home/ec2-user/Amazon-ML-Hackathon/dataset/student_resource/dataset/test"),
        REPO_ROOT / "data" / "student_resource" / "dataset" / "test",
    ]
    test_dir = None
    for p in candidates_test_dirs:
        if p.is_dir() and (p / "test_source1.tsv").is_file():
            test_dir = p
            break

    if test_dir is None:
        print("  [FAIL] Test dataset directory not found!")
        print(f"  Checked: {[str(p) for p in candidates_test_dirs]}")
        sys.exit(1)
    else:
        print(f"Test Dataset Dir  : {test_dir}")
        for s in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
            fp = test_dir / s
            if fp.is_file():
                mb = fp.stat().st_size / (1024 * 1024)
                print(f"  [OK] {s} ({mb:.1f} MB)")
            else:
                print(f"  [FAIL] Missing {s}!")
                sys.exit(1)

    # 4. Trained CatBoost Models
    candidates_model_dirs = []
    env_dir = os.environ.get("BER_MODELS_DIR")
    if env_dir:
        candidates_model_dirs.append(Path(env_dir))

    candidates_model_dirs.extend([
        REPO_ROOT / "data" / "student_resource" / "outputs" / "matching" / "models",
        Path("/home/ec2-user/Amazon-ML-Hackathon/data/student_resource/outputs/matching/models"),
        REPO_ROOT / "data" / "outputs" / "matching" / "models",
        Path("/home/ec2-user/Amazon-ML-Hackathon/data/outputs/matching/models"),
    ])
    models_dir = None
    for p in candidates_model_dirs:
        if p.is_dir() and len(list(p.glob("catboost_matcher_fold*.cbm"))) == 5:
            models_dir = p
            break

    if models_dir is None:
        print("  [FAIL] 5 trained CatBoost models not found!")
        print(f"  Checked: {[str(p) for p in candidates_model_dirs]}")
        sys.exit(1)
    else:
        print(f"Models Directory  : {models_dir}")
        for fold in range(1, 6):
            mf = models_dir / f"catboost_matcher_fold{fold}.cbm"
            if mf.is_file():
                mb = mf.stat().st_size / (1024 * 1024)
                print(f"  [OK] catboost_matcher_fold{fold}.cbm ({mb:.1f} MB)")
            else:
                print(f"  [FAIL] Missing {mf.name}!")
                sys.exit(1)

    # 5. Core modules
    required_modules = ["pandas", "numpy", "pyarrow", "catboost"]
    for mod in required_modules:
        try:
            __import__(mod)
            print(f"  [OK] Python module '{mod}' importable.")
        except ImportError:
            print(f"  [FAIL] Required Python module '{mod}' is not installed!")
            sys.exit(1)

    print("\n[OK] All prerequisites verified successfully!\n")


def run_command(cmd: list) -> int:
    """Executes a subprocess and streams output to stdout in real time."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):
        print(line, end="")
    proc.stdout.close()
    return proc.wait()


def main():
    parser = argparse.ArgumentParser(description="End-to-End Test/Inference Pipeline Orchestrator")
    parser.add_argument("--start-from-step", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Step number to start from (default: 1)")
    parser.add_argument("--step", type=int, default=None, choices=[1, 2, 3, 4],
                        help="Run only this specific step")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Matching decision threshold for Step 3 (default: 0.95)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify prerequisites and scripts without running")
    args = parser.parse_args()

    start_total = time.time()
    print("=" * 80)
    print("AMAZON ML HACKATHON: FINAL TEST INFERENCE PIPELINE")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Check prerequisites
    check_prerequisites()

    if args.dry_run:
        print("Dry run requested. Exiting without executing steps.")
        return 0

    steps_to_run = []
    for step in PIPELINE_STEPS:
        if args.step is not None:
            if step["step_num"] == args.step:
                steps_to_run.append(step)
        elif step["step_num"] >= args.start_from_step:
            steps_to_run.append(step)

    print("Steps scheduled for execution:")
    for step in steps_to_run:
        print(f"  Step {step['step_num']}: {step['name']}")
    print("-" * 80)

    step_times = {}

    for step in steps_to_run:
        s_num = step["step_num"]
        s_name = step["name"]
        s_script = step["script"]

        print("\n" + "#" * 80)
        print(f"# LAUNCHING STEP {s_num}: {s_name.upper()}")
        print(f"# Script: {s_script.name}")
        print(f"# Description: {step['description']}")
        print(f"# Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("#" * 80 + "\n")

        t0 = time.time()
        cmd = [sys.executable, "-u", str(s_script)]
        if s_num == 3 and args.threshold is not None:
            cmd.extend(["--threshold", str(args.threshold)])
        exit_code = run_command(cmd)

        elapsed_min = (time.time() - t0) / 60
        step_times[s_num] = elapsed_min

        if exit_code != 0:
            print("\n" + "!" * 80)
            print(f"! ERROR: Step {s_num} ({s_name}) failed with exit code {exit_code}!")
            print(f"! Elapsed time before failure: {elapsed_min:.2f} mins")
            print("!" * 80)
            sys.exit(exit_code)

        print(f"\n>>> Step {s_num} completed successfully in {elapsed_min:.2f} mins.")

    # Final summary
    total_elapsed = (time.time() - start_total) / 60
    print("\n" + "=" * 80)
    print("FULL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Execution Timing Summary:")
    for s_num, dur in step_times.items():
        step_meta = next(s for s in PIPELINE_STEPS if s["step_num"] == s_num)
        print(f"  Step {s_num} ({step_meta['name']}): {dur:.2f} mins")
    print(f"\nTotal Pipeline Elapsed Time: {total_elapsed:.2f} mins")

    # Deliverables verification
    out_dir = REPO_ROOT / "output"
    match_file = out_dir / "matching_results.tsv"
    cand_file = out_dir / "candidate_pairs.tsv"

    print("\nOfficial Submission Deliverables:")
    if match_file.is_file():
        mb = match_file.stat().st_size / (1024 * 1024)
        print(f"  [OK] {match_file} ({mb:.1f} MB)")
    else:
        print(f"  [FAIL] Missing {match_file}!")

    if cand_file.is_file():
        mb = cand_file.stat().st_size / (1024 * 1024)
        print(f"  [OK] {cand_file} ({mb:.1f} MB)")
    else:
        print(f"  [FAIL] Missing {cand_file}!")

    print("\nSubmission is ready for upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
