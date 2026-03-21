"""Integration tests validating precomputed pipeline output."""

import json
import math
import os
import pytest
from tests.conftest import PRECOMPUTED_DIR, REQUIRED_CANDIDATE_FIELDS


PATHOGEN_FILES = [
    "botrytis_cinerea",
    "fusarium_graminearum",
    "ralstonia_solanacearum",
    "xanthomonas_citri",
]


@pytest.mark.integration
class TestPrecomputedFiles:
    def test_all_4_files_exist(self):
        """All 4 pathogen result files exist."""
        for name in PATHOGEN_FILES:
            path = os.path.join(PRECOMPUTED_DIR, f"{name}.json")
            assert os.path.exists(path), f"Missing: {name}.json"

    def test_candidate_count_range(self, all_precomputed):
        """Each file has 150-170 candidates."""
        for name, data in all_precomputed.items():
            n = len(data["candidates"])
            assert 150 <= n <= 170, f"{name}: {n} candidates"

    def test_all_19_fields_present(self, all_precomputed):
        """Every candidate has all 19 required fields."""
        for name, data in all_precomputed.items():
            for i, c in enumerate(data["candidates"]):
                missing = set(REQUIRED_CANDIDATE_FIELDS) - set(c.keys())
                assert not missing, f"{name} candidate {i} missing: {missing}"

    def test_epinecidin_validation(self, precomputed_botrytis):
        """Botrytis results contain Epinecidin-1 and EPI-4; EPI-4 wins."""
        candidates = precomputed_botrytis["candidates"]
        epi1 = [c for c in candidates if c["name"] == "Epinecidin-1"]
        epi4 = [c for c in candidates if c["name"] == "EPI-4"]
        assert len(epi1) == 1, "Epinecidin-1 not found"
        assert len(epi4) == 1, "EPI-4 not found"
        assert epi4[0]["agriamp_score"] > epi1[0]["agriamp_score"]

    def test_cross_pathogen_same_metrics_keys(self, all_precomputed):
        """All 4 pathogen files have identical metrics key sets."""
        keys_list = [set(data["metrics"].keys()) for data in all_precomputed.values()]
        for keys in keys_list[1:]:
            assert keys == keys_list[0]

    def test_no_nan_in_candidates(self, all_precomputed):
        """No NaN in any numeric field across all candidates."""
        numeric_fields = [
            "agriamp_score", "amp_probability", "net_charge", "molecular_weight",
            "gravy", "hydrophobic_moment", "isoelectric_point", "instability_index",
            "aromaticity", "aliphatic_index", "boman_index", "toxicity_risk",
            "selectivity_score",
        ]
        for name, data in all_precomputed.items():
            for i, c in enumerate(data["candidates"]):
                for field in numeric_fields:
                    val = c[field]
                    assert not (isinstance(val, float) and math.isnan(val)), \
                        f"{name} candidate {i} {field} is NaN"
