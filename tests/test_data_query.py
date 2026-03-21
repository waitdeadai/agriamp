"""Unit tests for AMP database query (tools/data_query.py)."""

import pytest
from tools.data_query import (
    CURATED_ANTIFUNGAL_AMPS,
    PATHOGEN_KEYWORDS,
    DataQueryTool,
)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


@pytest.mark.unit
class TestCuratedDatabase:
    def test_all_valid_sequences(self):
        """All 27 curated AMP sequences contain only standard amino acids."""
        for name, seq, _, _, _ in CURATED_ANTIFUNGAL_AMPS:
            invalid = set(seq) - STANDARD_AA
            assert not invalid, f"{name} has invalid AAs: {invalid}"

    def test_all_positive_mic(self):
        """All MIC values are positive."""
        for name, _, _, _, mic in CURATED_ANTIFUNGAL_AMPS:
            assert mic > 0, f"{name} MIC={mic}"

    def test_no_duplicate_names(self):
        """No duplicate peptide names in curated database."""
        names = [name for name, _, _, _, _ in CURATED_ANTIFUNGAL_AMPS]
        assert len(names) == len(set(names))

    def test_count(self):
        """Database has 27 entries."""
        assert len(CURATED_ANTIFUNGAL_AMPS) == 27


@pytest.mark.unit
class TestDataQueryTool:
    def test_botrytis_finds_epinecidin(self):
        """Botrytis cinerea query returns both Epinecidin-1 and EPI-4."""
        tool = DataQueryTool()
        result = tool._execute(pathogen="Botrytis cinerea")
        assert result.status == "success"
        names = [amp["name"] for amp in result.data["curated_amps"]]
        assert "Epinecidin-1" in names
        assert "EPI-4" in names

    def test_all_pathogens_return_results(self):
        """All 4 pathogens produce non-empty curated_amps."""
        tool = DataQueryTool()
        for pathogen in PATHOGEN_KEYWORDS:
            result = tool._execute(pathogen=pathogen)
            assert result.status == "success"
            assert len(result.data["curated_amps"]) > 0, f"No results for {pathogen}"

    def test_seeds_max_10(self):
        """seed_sequences list never exceeds 10."""
        tool = DataQueryTool()
        result = tool._execute(pathogen="Botrytis cinerea")
        assert len(result.data["seed_sequences"]) <= 10

    def test_unknown_pathogen_fallback(self):
        """Unknown pathogen still returns results via fallback keywords."""
        tool = DataQueryTool()
        result = tool._execute(pathogen="Unknown pathogen XYZ")
        assert result.status == "success"
        assert len(result.data["curated_amps"]) > 0
