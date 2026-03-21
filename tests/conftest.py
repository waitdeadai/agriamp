"""Shared fixtures for AgriAMP test suite."""

import json
import os
import sys
import math
import pytest
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PRECOMPUTED_DIR = os.path.join(PROJECT_ROOT, "data", "precomputed")


@pytest.fixture
def known_peptides():
    """Well-characterized peptides from literature with known properties."""
    return {
        "Magainin 2": "GIGKFLHSAKKFGKAFVGEIMNS",
        "Epinecidin-1": "GFIFHIIKGLFHAGKMIHGLV",
        "EPI-4": "GFIFHIIKGLFHAGKMIHGLVK",
        "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "Indolicidin": "ILPWKWPWWPWRR",
        "Aurein 1.2": "GLFDIIKKIAESF",
        "Temporin A": "FLPLIGRVLSGIL",
        "Melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    }


@pytest.fixture
def precomputed_botrytis():
    """Load precomputed Botrytis cinerea pipeline results."""
    path = os.path.join(PRECOMPUTED_DIR, "botrytis_cinerea.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def all_precomputed():
    """Load all 4 precomputed pathogen results."""
    results = {}
    for fname in ["botrytis_cinerea", "fusarium_graminearum",
                   "ralstonia_solanacearum", "xanthomonas_citri"]:
        path = os.path.join(PRECOMPUTED_DIR, f"{fname}.json")
        with open(path, "r", encoding="utf-8") as f:
            results[fname] = json.load(f)
    return results


@pytest.fixture
def edge_sequences():
    """Edge-case sequences for stress testing."""
    return {
        "poly_K": "KKKKKKKKKK",
        "poly_D": "DDDDDDDDDD",
        "poly_A": "AAAAAAAAAA",
        "alternating_KL": "KLKLKLKLKL",
        "cysteine_rich": "CCCCCCCCCC",
        "all_W": "WWWWWWWWWW",
        "minimum_3": "KKK",
        "hemolytic_motif": "AAWKKWAAKL",
        "long_100": "A" * 50 + "K" * 50,
    }


@pytest.fixture
def standard_aa():
    return set("ACDEFGHIKLMNPQRSTVWY")


def compute_agriamp_score(amp_prob, charge, hm, instability, tox_risk):
    """Replicate the composite scoring formula from agent.py:296-307."""
    charge_norm = min(max(charge / 6, 0), 1)
    hm_norm = min(max(hm / 0.8, 0), 1)
    stability_norm = max(1 - instability / 100, 0)
    return (
        0.35 * amp_prob
        + 0.25 * charge_norm
        + 0.20 * hm_norm
        + 0.10 * stability_norm
        + 0.10 * (1 - tox_risk)
    )


REQUIRED_CANDIDATE_FIELDS = [
    "sequence", "length", "agriamp_score", "amp_probability",
    "net_charge", "molecular_weight", "gravy", "hydrophobic_moment",
    "isoelectric_point", "instability_index", "aromaticity",
    "aliphatic_index", "boman_index", "toxicity_risk",
    "selectivity_score", "passed_toxicity", "tox_flags", "origin", "name",
]
