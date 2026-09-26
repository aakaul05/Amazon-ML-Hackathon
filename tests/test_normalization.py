"""
tests/test_normalization.py
===========================
Unit tests and invariant assertions for Task 4 Normalization.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure src is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from business_entity_resolution.preprocessing.normalization import (
    basic_clean_string,
    strip_accents_and_diacritics,
    strip_legal_suffix,
    normalize_series_fast,
    LEGAL_SUFFIX_LIST,
)


class TestBasicNormalization:
    """Tests for basic cleaning (accents, casing, punctuation, whitespace, digits)."""

    def test_accents_and_diacritics(self):
        # NFKD + combining marks removal
        assert strip_accents_and_diacritics("Payne Énterprises") == "Payne Enterprises"
        assert strip_accents_and_diacritics("Société Générale") == "Societe Generale"
        assert strip_accents_and_diacritics("München Über") == "Munchen Uber"
        assert basic_clean_string("Payne Énterprises") == "payne enterprises"

    def test_casing(self):
        assert basic_clean_string("ABC TECHNOLOGIES") == "abc technologies"
        assert basic_clean_string("mAuRe WiLlIaMs") == "maure williams"

    def test_punctuation_replacement(self):
        assert basic_clean_string("Obsidian,-LLC") == "obsidian llc"
        assert basic_clean_string("Hendricks & Flowers, Inc.") == "hendricks flowers inc"
        assert basic_clean_string("Co./Corp.") == "co corp"

    def test_whitespace_collapsing(self):
        assert basic_clean_string("   A   B     C   ") == "a b c"
        assert basic_clean_string("\t\nPayne  \t Enterprises \n") == "payne enterprises"

    def test_digits_and_address_alphanumerics_preserved(self):
        # Addresses contain critical numeric tokens
        addr = "12A Main St., Suite #400, Apt 5B, 90210"
        cleaned = basic_clean_string(addr)
        assert "12a" in cleaned
        assert "400" in cleaned
        assert "5b" in cleaned
        assert "90210" in cleaned
        assert cleaned == "12a main st suite 400 apt 5b 90210"

    def test_null_and_empty_handling(self):
        assert basic_clean_string(None) == ""
        assert basic_clean_string("") == ""
        assert basic_clean_string("    ") == ""
        assert basic_clean_string(float("nan")) == ""
        assert basic_clean_string(np.nan) == ""

    def test_non_string_conversion(self):
        assert basic_clean_string(12345) == "12345"
        assert basic_clean_string(42.0) == "42 0"


class TestLegalSuffixNormalization:
    """Tests for strict terminal legal suffix stripping and invariant preservation."""

    def test_terminal_single_word_suffix(self):
        assert strip_legal_suffix("abc ltd") == "abc"
        assert strip_legal_suffix("abc inc") == "abc"
        assert strip_legal_suffix("abc gmbh") == "abc"
        assert strip_legal_suffix("abc llc") == "abc"

    def test_compound_suffixes(self):
        assert strip_legal_suffix("abc private limited") == "abc"
        assert strip_legal_suffix("abc pvt ltd") == "abc"
        assert strip_legal_suffix("abc co ltd") == "abc"
        assert strip_legal_suffix("abc pty ltd") == "abc"

    def test_chained_suffixes(self):
        # "ABC Company Limited" -> strips "limited" then "company" -> "abc"
        assert strip_legal_suffix("abc company limited") == "abc"
        # "ABC Holdings Inc" -> strips "inc", leaves "abc holdings" because "holdings" is descriptive
        assert strip_legal_suffix("abc holdings inc") == "abc holdings"

    def test_suffix_in_middle_or_start_not_stripped(self):
        # Suffix must be at the end of the string
        assert strip_legal_suffix("limited brands") == "limited brands"
        assert strip_legal_suffix("company store inc") == "company store"
        assert strip_legal_suffix("inc logistics llc") == "inc logistics"
        assert strip_legal_suffix("general electric co") == "general electric"

    def test_descriptive_words_strictly_preserved(self):
        """
        CRITICAL INVARIANT:
        Descriptive business nouns must NOT be stripped under any circumstance:
        hotel, enterprises, partners, services, solutions, group, holdings, international.
        """
        descriptive_terms = [
            "hotel",
            "enterprises",
            "partners",
            "services",
            "solutions",
            "group",
            "holdings",
            "international",
        ]
        # Ensure none of these descriptive tokens exist in the suffix list
        for term in descriptive_terms:
            assert term not in LEGAL_SUFFIX_LIST, f"Descriptive term '{term}' illegally present in LEGAL_SUFFIX_LIST"

        # Explicit test cases from specifications
        assert strip_legal_suffix(basic_clean_string("Hotel Enterprises Ltd")) == "hotel enterprises"
        assert strip_legal_suffix(basic_clean_string("Hotel Enterprises")) == "hotel enterprises"
        assert strip_legal_suffix(basic_clean_string("Chordia & Partners Company")) == "chordia partners"
        assert strip_legal_suffix(basic_clean_string("Global Logistics Group Inc")) == "global logistics group"
        assert strip_legal_suffix(basic_clean_string("International Services LLC")) == "international services"

    def test_safety_guard_names_never_empty(self):
        # Short / standalone legal names must not be erased
        assert strip_legal_suffix("ltd") == "ltd"
        assert strip_legal_suffix("inc") == "inc"
        assert strip_legal_suffix("co") == "co"
        assert strip_legal_suffix("gmbh") == "gmbh"
        assert strip_legal_suffix("a ltd") == "a ltd"
        assert strip_legal_suffix("b inc") == "b inc"


class TestSeriesNormalizationFast:
    """Test vectorized normalization using unique mapping."""

    def test_series_mapping_consistency(self):
        s = pd.Series([
            "Payne Énterprises",
            "Hotel Enterprises Ltd",
            "ABC Private Limited",
            None,
            np.nan,
            "Payne Énterprises",  # duplicate
        ])
        norm, legal = normalize_series_fast(s, is_name=True)

        assert len(norm) == len(s)
        assert len(legal) == len(s)

        assert norm[0] == "payne enterprises"
        assert norm[5] == "payne enterprises"
        assert norm[3] == ""
        assert norm[4] == ""

        assert legal[0] == "payne enterprises"
        assert legal[1] == "hotel enterprises"
        assert legal[2] == "abc"
        assert legal[3] == ""
        assert legal[4] == ""
