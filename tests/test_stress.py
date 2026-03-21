"""Stress tests: edge cases, adversarial inputs, boundary conditions."""

import math
import pytest
import numpy as np
from tools.properties import (
    compute_all_properties,
    compute_net_charge,
    compute_hydrophobic_moment,
    compute_molecular_weight,
    AA_MW,
)
from tools.toxicity import compute_toxicity_score
from tools.generator import GeneratorTool

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")


@pytest.mark.stress
class TestHomopolymers:
    def test_all_20_homopolymers_compute(self):
        """compute_all_properties works for each AA as a 10-mer."""
        for aa in STANDARD_AAS:
            seq = aa * 10
            result = compute_all_properties(seq)
            assert result is not None, f"Failed for poly-{aa}"

    def test_no_nan_inf_any_homopolymer(self):
        """No NaN or Inf in any property for any homopolymer."""
        for aa in STANDARD_AAS:
            seq = aa * 10
            result = compute_all_properties(seq)
            for key, val in result.items():
                if isinstance(val, (int, float)):
                    assert math.isfinite(val), f"poly-{aa}.{key} = {val}"


@pytest.mark.stress
class TestExtremeLengths:
    def test_200_residues(self):
        """200-residue peptide computes correctly."""
        seq = "KKLLAAGGII" * 20  # 200 aa
        result = compute_all_properties(seq)
        assert result is not None
        assert result["length"] == 200

    def test_minimum_3(self):
        """3-residue peptide is the minimum that returns results."""
        assert compute_all_properties("KKK") is not None
        assert compute_all_properties("KK") is None


@pytest.mark.stress
class TestNonStandardAA:
    def test_cleaned_correctly(self):
        """Non-standard AAs (B, X, Z, J, O, U) are removed during cleaning."""
        result = compute_all_properties("KKBXZJOUKK")
        assert result is not None
        # Only K's survive cleaning
        assert result["sequence"] == "KKKK"


@pytest.mark.stress
class TestToxicityEdgeCases:
    def test_poly_k_high_charge_flag(self):
        """Poly-K (10 residues, charge ~+10) triggers very high charge flag."""
        props = {"gravy": -3.9, "net_charge_ph7": 10.0, "boman_index": 2.71,
                 "hydrophobic_moment": 0.0}
        score, flags = compute_toxicity_score("K" * 10, props)
        assert any("cytotoxicity" in f.lower() for f in flags)

    def test_all_triggers_capped(self):
        """Worst-case peptide triggering everything: score capped at 1.0."""
        # Long, Cys-rich, WxxW, high GRAVY, high charge, high Boman
        seq = "C" * 8 + "WKKW" + "A" * 42  # 54 aa, >12% C, WxxW, >50 aa
        props = {"gravy": 0.8, "net_charge_ph7": 12.0, "boman_index": 4.0,
                 "hydrophobic_moment": 0.1}
        score, flags = compute_toxicity_score(seq, props)
        assert score == 1.0

    def test_boundary_gravy_030(self):
        """GRAVY exactly at 0.3 should NOT trigger moderate flag (threshold is >0.3)."""
        score, flags = compute_toxicity_score(
            "AAAAAAAAAA",
            {"gravy": 0.3, "net_charge_ph7": 4.0, "boman_index": 1.0,
             "hydrophobic_moment": 0.5},
        )
        assert not any("hydrophobicity" in f.lower() for f in flags)

    def test_boundary_gravy_050(self):
        """GRAVY exactly at 0.5 should NOT trigger high flag (threshold is >0.5)."""
        score, flags = compute_toxicity_score(
            "AAAAAAAAAA",
            {"gravy": 0.5, "net_charge_ph7": 4.0, "boman_index": 1.0,
             "hydrophobic_moment": 0.5},
        )
        # 0.5 is not > 0.5, so high flag should NOT trigger
        assert not any("host cell lysis" in f.lower() for f in flags)

    def test_boundary_charge_exact_9(self):
        """Charge exactly 9.0 should NOT trigger very-high-charge flag (threshold >9)."""
        score, flags = compute_toxicity_score(
            "AAAAAAAAAA",
            {"gravy": 0.0, "net_charge_ph7": 9.0, "boman_index": 1.0,
             "hydrophobic_moment": 0.5},
        )
        assert not any("cytotoxicity" in f.lower() for f in flags)


@pytest.mark.stress
class TestConservativeSubstitution:
    def test_k_to_r_similar_charge(self):
        """K→R substitution (both cationic): charge difference < 0.5."""
        seq_k = "GIGKFLHSAKKFGKAFVGEIMNS"
        seq_r = "GIGRFLHSAKKFGKAFVGEIMNS"  # K4→R4
        charge_k = compute_net_charge(seq_k, 7.0)
        charge_r = compute_net_charge(seq_r, 7.0)
        assert abs(charge_k - charge_r) < 0.5


@pytest.mark.stress
class TestHydrophobicMomentOrder:
    def test_scrambled_different_hm(self):
        """Scrambled Magainin 2 has different HM (sequence order matters)."""
        original = "GIGKFLHSAKKFGKAFVGEIMNS"
        scrambled = "SGKGFAIVKEMFNHKGALKSIG"  # manually reordered
        hm_orig = compute_hydrophobic_moment(original)
        hm_scram = compute_hydrophobic_moment(scrambled)
        assert hm_orig != hm_scram

    def test_charge_monotonicity_adding_K(self):
        """Adding a lysine always increases charge at pH 7."""
        base = "AAAAAA"
        for i in range(1, 6):
            seq = base + "K" * i
            charge_i = compute_net_charge(seq, 7.0)
            seq_more = base + "K" * (i + 1)
            charge_more = compute_net_charge(seq_more, 7.0)
            assert charge_more > charge_i

    def test_truncation_mw_proportional(self):
        """Removing 2 N-terminal residues changes MW by ~2 AA weights."""
        seq = "GIGKFLHSAKKFGKAFVGEIMNS"
        truncated = seq[2:]
        mw_full = compute_molecular_weight(seq)
        mw_trunc = compute_molecular_weight(truncated)
        # MW difference ≈ sum of removed AA weights - 2 waters (2 peptide bonds)
        expected_diff = AA_MW.get(seq[0], 0) + AA_MW.get(seq[1], 0) - 2 * 18.02
        actual_diff = mw_full - mw_trunc
        assert abs(actual_diff - expected_diff) < 1.0


@pytest.mark.stress
class TestGeneratorEdgeCases:
    def test_cationic_seed(self):
        """Poly-K seed: point mutations replace K with L/W/F/I (not K itself)."""
        gen = GeneratorTool()
        result = gen._execute(seed_sequences=["KKKKKKKKKK"], max_total=50)
        assert result.status == "success"
        point_muts = [v for v in result.data["variants"]
                      if v["strategy"] == "point_mutation"]
        for v in point_muts:
            # Should replace K with non-K enhancing AAs
            for a, b in zip(v["parent"], v["sequence"]):
                if a != b:
                    assert b != "K"
