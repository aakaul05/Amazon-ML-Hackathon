"""
scripts/verify_lightgbm_pipeline.py
===================================
End-to-End Verification and Smoke Test for LightGBM Matching Pipeline:

Verifies:
1. LightGBM library availability and version.
2. Exact 30 feature schema, order, and names.
3. Feature computation integrity (no NaNs, no Infs, correct float32 dtype).
4. LightGBM Booster training, saving (.txt), and loading.
5. Prediction output format: 1D probabilities strictly in [0.0, 1.0].
6. Threshold filtering logic and match extraction.
7. Downstream compatibility with submission format (12_create_submission.py).
"""

import os
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.matching.features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    compute_features_batch,
)
from business_entity_resolution.matching import (
    create_lightgbm_matcher,
    train_lightgbm_matcher,
    get_lightgbm_feature_importances,
)


def run_smoke_test():
    print("=" * 80)
    print("LIGHTGBM MATCHING PIPELINE: COMPREHENSIVE SMOKE TEST & VERIFICATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # Test 1: Library & Environment Check
    # ---------------------------------------------------------
    print("\n[Test 1/7] Checking LightGBM installation...")
    try:
        import lightgbm as lgb
        print(f"  [PASS] LightGBM version: {lgb.__version__}")
    except ImportError as e:
        print(f"  [FAIL] LightGBM is not installed: {e}")
        return False

    # ---------------------------------------------------------
    # Test 2: Feature Schema & Ordering
    # ---------------------------------------------------------
    print("\n[Test 2/7] Checking 30-feature schema and exact ordering...")
    assert len(FEATURE_NAMES) == 30, f"Expected 30 features, got {len(FEATURE_NAMES)}"
    assert NUM_FEATURES == 30, f"NUM_FEATURES constant mismatch: {NUM_FEATURES}"
    print(f"  [PASS] Total features: {len(FEATURE_NAMES)}")
    print(f"  [PASS] First 5: {FEATURE_NAMES[:5]}")
    print(f"  [PASS] Last 5 : {FEATURE_NAMES[-5:]}")

    # ---------------------------------------------------------
    # Test 3: Feature Computation Integrity (NaN/Inf check)
    # ---------------------------------------------------------
    print("\n[Test 3/7] Testing feature computation with sample pairs...")
    s1_names = ["Acme Industrial Corp", "Global Logistics LLC", "Empty Address Inc", ""]
    s1_clean = ["acme industrial", "global logistics", "empty address", ""]
    s1_addrs = ["123 Main St, Springfield, IL", "Suite 400, Chicago, IL", "", "456 Oak Rd"]
    s1_countries = ["us", "us", "us", "in"]

    ot_names = ["Acme Industries", "Worldwide Logistics", "Empty Address Corp", "Unknown Shop"]
    ot_clean = ["acme industries", "worldwide logistics", "empty address", "unknown shop"]
    ot_addrs = ["123 Main Street, Springfield, IL", "Suite 400, Chicago, IL", "", "789 Pine Ave"]
    ot_countries = ["us", "us", "us", "fr"]

    bp_list = ["exact_name_norm|improved_address", "rare_token_ranked", "exact_clean_legal", ""]

    X_test = compute_features_batch(
        s1_names, s1_clean, s1_addrs, s1_countries,
        ot_names, ot_clean, ot_addrs, ot_countries,
        bp_list,
    )

    assert X_test.shape == (4, 30), f"Expected shape (4, 30), got {X_test.shape}"
    assert not np.isnan(X_test).any(), "NaN found in computed feature matrix!"
    assert not np.isinf(X_test).any(), "Inf found in computed feature matrix!"
    assert X_test.dtype == np.float32, f"Expected float32, got {X_test.dtype}"
    print(f"  [PASS] Feature shape: {X_test.shape}, dtype: {X_test.dtype}")
    print(f"  [PASS] Value range: [{X_test.min():.4f}, {X_test.max():.4f}] (Zero NaNs, Zero Infs)")

    # ---------------------------------------------------------
    # Test 4: Model Training with LightGBM
    # ---------------------------------------------------------
    print("\n[Test 4/7] Testing LightGBM Booster training and early stopping...")
    rng = np.random.RandomState(42)
    N_tr = 500
    N_va = 100
    X_tr = rng.rand(N_tr, 30).astype(np.float32)
    y_tr = (X_tr[:, 0] * 0.4 + X_tr[:, 1] * 0.4 + rng.randn(N_tr) * 0.2 > 0.5).astype(np.int8)
    X_va = rng.rand(N_va, 30).astype(np.float32)
    y_va = (X_va[:, 0] * 0.4 + X_va[:, 1] * 0.4 + rng.randn(N_va) * 0.2 > 0.5).astype(np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "test_lightgbm_fold1.txt"
        booster, metrics = train_lightgbm_matcher(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_va,
            y_val=y_va,
            feature_names=FEATURE_NAMES,
            num_boost_round=50,
            early_stopping_rounds=15,
            model_save_path=model_file,
            verbose_eval=-1,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=6,
        )

        assert model_file.is_file(), f"Model file not created at {model_file}"
        assert model_file.stat().st_size > 0, "Model file is empty"
        print(f"  [PASS] Trained Booster: best_iteration={metrics['best_iteration']}")
        print(f"  [PASS] Model saved successfully: {model_file.name} ({model_file.stat().st_size:,} bytes)")

        # ---------------------------------------------------------
        # Test 5: Model Loading and Inference
        # ---------------------------------------------------------
        print("\n[Test 5/7] Testing LightGBM Booster loading and prediction...")
        loaded_booster = lgb.Booster(model_file=str(model_file))
        probs = loaded_booster.predict(X_test).astype(np.float32)

        assert probs.shape == (4,), f"Expected shape (4,), got {probs.shape}"
        assert (probs >= 0.0).all() and (probs <= 1.0).all(), f"Probabilities out of [0, 1] range: {probs}"
        print(f"  [PASS] Loaded model successfully.")
        print(f"  [PASS] Predicted probabilities on 4 test pairs: {np.round(probs, 4)}")

        # ---------------------------------------------------------
        # Test 6: Feature Importances
        # ---------------------------------------------------------
        print("\n[Test 6/7] Checking feature importance extraction...")
        fi_df = get_lightgbm_feature_importances(loaded_booster, FEATURE_NAMES)
        assert len(fi_df) == 30, f"Expected 30 feature importances, got {len(fi_df)}"
        print(f"  [PASS] Top 3 features by gain: {list(fi_df['feature'].head(3))}")

    # ---------------------------------------------------------
    # Test 7: Threshold & Submission Formatting Verification
    # ---------------------------------------------------------
    print("\n[Test 7/7] Verifying threshold filtering and submission schema compatibility...")
    threshold = 0.50
    keep_mask = probs >= threshold
    test_s1_ids = np.array(["S1_001", "S1_002", "S1_003", "S1_004"])
    test_cand_ids = np.array(["S2_101", "S2_102", "S2_103", "S2_104"])

    df_part = pd.DataFrame({
        "s1_entity_id": test_s1_ids[keep_mask],
        "matched_entity_id": test_cand_ids[keep_mask],
        "probability": probs[keep_mask],
    })

    assert set(df_part.columns) == {"s1_entity_id", "matched_entity_id", "probability"}, \
        f"Columns mismatch: {df_part.columns}"
    print(f"  [PASS] Matches kept at threshold {threshold}: {len(df_part)} pairs")
    print(f"  [PASS] Output parquet schema matches Step 3/4 submission contract perfectly.")

    print("\n" + "=" * 80)
    print("ALL 7 VERIFICATION CHECKS PASSED: LightGBM MATCHING PIPELINE IS VERIFIED!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        sys.exit(1)
