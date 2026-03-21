"""Unit tests for toxicity scoring (tools/toxicity.py)."""

import pytest
from tools.toxicity import compute_toxicity_score, compute_selectivity_estimate, ToxicityTool


def _make_props(**overrides):
    """Create a property dict with safe defaults, overriding as needed."""
    defaults = {
        "gravy": 0.0,
        "net_charge_ph7": 4.0,
        "hydrophobic_moment": 0.5,
        "boman_index": 1.0,
    }
    defaults.update(overrides)
    return defaults


# --- Individual Flag Tests ---

@pytest.mark.unit
class TestToxicityFlags:
    def test_ideal_peptide_zero_risk(self):
        """A peptide with all safe properties scores 0.0 with no flags."""
        seq = "KKLLKKLLKK"  # 10 aa, no Cys, no WxxW
        props = _make_props(gravy=0.1, net_charge_ph7=4.0, boman_index=1.0)
        score, flags = compute_toxicity_score(seq, props)
        assert score == 0.0
        assert flags == []

    def test_high_gravy_flag(self):
        """GRAVY > 0.5 triggers +0.25 hydrophobicity flag."""
        score, flags = compute_toxicity_score("AAAAAAAAAA", _make_props(gravy=0.6))
        assert score >= 0.25
        assert any("hydrophobicity" in f.lower() for f in flags)

    def test_moderate_gravy_flag(self):
        """GRAVY 0.3-0.5 triggers +0.10 moderate flag."""
        score, flags = compute_toxicity_score("AAAAAAAAAA", _make_props(gravy=0.35))
        assert score >= 0.10
        assert any("hydrophobicity" in f.lower() for f in flags)

    def test_very_high_charge_flag(self):
        """Charge > 9 triggers +0.25 cytotoxicity flag."""
        score, flags = compute_toxicity_score("AAAAAAAAAA", _make_props(net_charge_ph7=10.0))
        assert score >= 0.25
        assert any("cytotoxicity" in f.lower() for f in flags)

    def test_long_peptide_flag(self):
        """Length > 50 triggers +0.15 immunogenicity flag."""
        long_seq = "A" * 55
        score, flags = compute_toxicity_score(long_seq, _make_props())
        assert score >= 0.15
        assert any("immunogenicity" in f.lower() for f in flags)

    def test_cysteine_rich_flag(self):
        """Cysteine > 12% triggers +0.15 disulfide flag."""
        seq = "CCAAAAAAA"  # 2/9 = 22%
        score, flags = compute_toxicity_score(seq, _make_props())
        assert score >= 0.15
        assert any("disulfide" in f.lower() for f in flags)

    def test_hemolytic_wxxw(self):
        """WxxW motif triggers +0.20 hemolytic flag."""
        seq = "AAWKKWAAKL"
        score, flags = compute_toxicity_score(seq, _make_props())
        assert score >= 0.20
        assert any("hemolytic" in f.lower() for f in flags)

    def test_hemolytic_wxxxw(self):
        """WxxxW motif also matches."""
        seq = "AAWKKKWAAK"
        score, flags = compute_toxicity_score(seq, _make_props())
        assert any("hemolytic" in f.lower() for f in flags)

    def test_low_charge_flag(self):
        """Charge < 1 triggers +0.15 poor selectivity flag."""
        score, flags = compute_toxicity_score("AAAAAAAAAA", _make_props(net_charge_ph7=0.5))
        assert score >= 0.15
        assert any("selectivity" in f.lower() for f in flags)

    def test_high_boman_flag(self):
        """Boman > 2.5 triggers +0.10 host protein binding flag."""
        score, flags = compute_toxicity_score("AAAAAAAAAA", _make_props(boman_index=3.0))
        assert score >= 0.10
        assert any("protein" in f.lower() for f in flags)

    def test_score_capped_at_one(self):
        """Even with all flags active, score cannot exceed 1.0."""
        # Trigger everything: high GRAVY, high charge, long, Cys-rich, WxxW, high Boman
        seq = "C" * 10 + "WKKW" + "A" * 40  # >50 aa, >12% Cys, WxxW
        props = _make_props(gravy=0.8, net_charge_ph7=12.0, boman_index=4.0)
        score, flags = compute_toxicity_score(seq, props)
        assert score <= 1.0
        assert len(flags) >= 4  # multiple flags triggered


# --- Selectivity ---

@pytest.mark.unit
class TestSelectivity:
    def test_optimal_profile(self):
        """Charge 4, GRAVY 0.1, HM 0.6 → selectivity near 1.0."""
        sel = compute_selectivity_estimate(
            {"net_charge_ph7": 4.0, "gravy": 0.1, "hydrophobic_moment": 0.6}
        )
        assert sel["selectivity_score"] >= 0.9

    def test_poor_profile(self):
        """Charge 0, GRAVY 1.5, HM 0.1 → low selectivity."""
        sel = compute_selectivity_estimate(
            {"net_charge_ph7": 0.0, "gravy": 1.5, "hydrophobic_moment": 0.1}
        )
        assert sel["selectivity_score"] <= 0.4

    def test_returns_all_components(self):
        """Result contains all 4 selectivity keys."""
        sel = compute_selectivity_estimate(
            {"net_charge_ph7": 4.0, "gravy": 0.0, "hydrophobic_moment": 0.5}
        )
        assert "selectivity_score" in sel
        assert "charge_selectivity" in sel
        assert "hydrophobicity_selectivity" in sel
        assert "amphipathicity_selectivity" in sel


# --- ToxicityTool Integration ---

@pytest.mark.unit
class TestToxicityToolIntegration:
    def test_execute_returns_correct_structure(self):
        """_execute with 3 sequences returns proper result structure."""
        tool = ToxicityTool()
        seqs = ["KKLLKKLLKK", "CCWKKWCCCC", "AAAAAAAAAA"]
        props_list = [
            _make_props(gravy=0.0, net_charge_ph7=4.0),
            _make_props(gravy=0.6, net_charge_ph7=2.0),
            _make_props(gravy=-0.5, net_charge_ph7=0.5),
        ]
        result = tool._execute(sequences=seqs, properties_list=props_list)
        assert result.status == "success"
        assert len(result.data["toxicity_results"]) == 3
        assert "n_passed" in result.data
        assert "n_flagged" in result.data
        assert result.data["n_passed"] + result.data["n_flagged"] == 3
