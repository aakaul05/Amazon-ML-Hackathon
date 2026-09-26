"""
business_entity_resolution.preprocessing.normalization
======================================================
Reusable cleaning and normalization utilities for multi-source entity resolution.
"""

import re
import unicodedata
from typing import Tuple, Optional
import pandas as pd


def strip_accents_and_diacritics(text: str) -> str:
    """
    Decomposes characters (NFKD) and strips non-spacing combining marks (Mn).
    Example: "Payne Énterprises" -> "Payne Enterprises"
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def basic_clean_string(text: str) -> str:
    """
    1. Unicode accent decomposition (NFKD)
    2. Lowercase
    3. Punctuation replaced with space (preserving alphanumeric characters)
    4. Whitespace collapsed and trimmed
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = strip_accents_and_diacritics(str(text))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


LEGAL_SUFFIX_LIST = [
    # Compound suffixes
    "pvt ltd",
    "private limited",
    "co ltd",
    "company limited",
    "pty ltd",
    "proprietory limited",
    "corp inc",
    "llc inc",
    "sa de cv",
    "s de rl de cv",
    "sp z o o",
    
    # Common English suffixes
    "inc",
    "incorporated",
    "corporation",
    "corp",
    "limited",
    "ltd",
    "llc",
    "llp",
    "plc",
    "lp",
    "pvt",
    "private",
    "co",
    "company",
    
    # European suffixes
    "gmbh",
    "ag",
    "kgaa",
    "sarl",
    "sas",
    "sasu",
    "sa",
    "srl",
    "spa",
    "snc",
    "sl",
    "slne",
    "bv",
    "nv",
    "vof",
    "ab",
    "aps",
    "as",
    "oy",
    "oyj",
    
    # Asian transliterations
    "kk",
    "kabushiki kaisha",
    "yk",
    "yugen kaisha",
]

SUFFIX_PATTERN = re.compile(
    r"\s+(?:" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIX_LIST, key=len, reverse=True)) + r")$",
    flags=re.IGNORECASE
)


def strip_legal_suffix(text: str) -> str:
    """
    Recursively strips legal corporate designations strictly from the END of the normalized business name.
    Does not strip descriptive tokens (e.g. 'enterprises', 'partners', 'services').
    """
    if not text:
        return ""
    
    current = text
    for _ in range(3):
        new_text = SUFFIX_PATTERN.sub("", current).strip()
        if new_text == current or len(new_text) < 2:
            break
        current = new_text
        
    return current


def normalize_series_fast(series: pd.Series, is_name: bool = False) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Applies normalization across unique values to avoid recomputing duplicates across millions of rows.
    """
    unique_vals = series.dropna().unique()
    cleaned_map = {val: basic_clean_string(val) for val in unique_vals}
    basic_series = series.map(cleaned_map).fillna("")
    
    if is_name:
        unique_cleaned = set(cleaned_map.values())
        legal_map = {val: strip_legal_suffix(val) for val in unique_cleaned}
        legal_series = basic_series.map(legal_map).fillna("")
        return basic_series, legal_series
    
    return basic_series, None


def normalize_source_df(df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    """
    Applies full name, address, and country normalization pipeline to a source DataFrame.
    """
    if source_name:
        print(f"Normalizing {source_name}...")
    df["name_norm"], df["name_clean_legal"] = normalize_series_fast(
        df["business_name"], is_name=True
    )
    df["address_norm"], _ = normalize_series_fast(
        df["business_address"], is_name=False
    )
    df["country_norm"], _ = normalize_series_fast(
        df["country"], is_name=False
    )
    return df


def load_normalized_or_compute(train_dir, repo_root):
    """
    Loads pre-normalized Parquet cache if available.
    Falls back to raw TSV loading + normalization if cache is missing.
    """
    from pathlib import Path
    train_dir = Path(train_dir)
    repo_root = Path(repo_root)

    cache_dir = repo_root / "data" / "outputs" / "normalized_cache"
    cache_files = [
        cache_dir / "s1_normalized.parquet",
        cache_dir / "s2_normalized.parquet",
        cache_dir / "s3_normalized.parquet",
    ]

    if all(f.exists() for f in cache_files):
        print("Loading pre-normalized Parquet cache (fast path)...")
        s1 = pd.read_parquet(cache_files[0])
        s2 = pd.read_parquet(cache_files[1])
        s3 = pd.read_parquet(cache_files[2])
        return s1, s2, s3

    # Fallback: load raw, normalize one-by-one and persist to cache to prevent OOM
    print("Cache not found. Normalizing one-by-one to Parquet cache (low-memory mode)...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    import gc

    cols = ["entity_id", "business_name", "business_address", "country"]
    for src_file, parquet_file, name in [
        ("train_source1.tsv", cache_files[0], "S1"),
        ("train_source2.tsv", cache_files[1], "S2"),
        ("train_source3.tsv", cache_files[2], "S3"),
    ]:
        if not parquet_file.exists():
            print(f"  Loading & Normalizing {name} ({src_file})...")
            df = pd.read_csv(train_dir / src_file, sep="\t", usecols=cols, dtype=str)
            df["name_norm"], df["name_clean_legal"] = normalize_series_fast(df["business_name"], is_name=True)
            df["address_norm"], _ = normalize_series_fast(df["business_address"], is_name=False)
            df["country_norm"], _ = normalize_series_fast(df["country"], is_name=False)
            df.to_parquet(parquet_file, index=False)
            print(f"  Saved {name} cache -> {parquet_file}")
            del df
            gc.collect()

    print("Loading normalized Parquet files into memory...")
    s1 = pd.read_parquet(cache_files[0])
    s2 = pd.read_parquet(cache_files[1])
    s3 = pd.read_parquet(cache_files[2])
    return s1, s2, s3
