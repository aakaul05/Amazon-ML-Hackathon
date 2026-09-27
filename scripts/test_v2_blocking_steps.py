"""
scripts/test_v2_blocking_steps.py
=================================
Incremental evaluation of V2 blocking improvements on 10k validation set:
1. Baseline V1 blocking (original passes and parameters)
2. Improvement 1: Deterministic ranked capping
3. Improvement 2: First-token and last-token blocking
4. Improvement 3: Improved address component blocking
5. Improvement 4: Partitioned TF-IDF coverage (full population)
"""

from pathlib import Path
import os
import sys
import json
import time
import gc
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.blocking.candidate_generator import _double_metaphone_simple

OUTPUT_DIR = REPO_ROOT / "data" / "student_resource" / "outputs" / "validation"
NORMALIZED_DIR = REPO_ROOT / "data" / "outputs" / "normalized_cache"

RANDOM_SEED = 42

# General business stopwords (avoid explosive first/last tokens)
NAME_STOPWORDS = {
    "the", "and", "for", "group", "services", "company", "international",
    "global", "solutions", "management", "systems", "technologies",
    "consulting", "enterprises", "partners", "associates", "holdings",
    "industries", "products", "national", "general", "american", "india",
    "business", "financial", "capital", "investments", "properties",
    "construction", "development", "insurance", "marketing", "trading",
    "logistics", "communications", "engineering", "electric", "energy",
    "first", "new", "all", "united", "central", "standard", "premier",
}

def load_validation_data():
    val_s1 = pd.read_parquet(OUTPUT_DIR / "val_s1_10k.parquet")
    with open(OUTPUT_DIR / "val_gt_10k.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_s2 = {(s1, m) for s1, matches in gt_data["s2"].items() for m in matches}
    gt_s3 = {(s1, m) for s1, matches in gt_data["s3"].items() for m in matches}
    return val_s1, gt_s2, gt_s3, gt_data

print("Loaded validation helper.")
