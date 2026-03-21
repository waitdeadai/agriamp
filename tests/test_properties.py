"""Unit tests for biochemical property calculations (tools/properties.py)."""

import math
import pytest
from tools.properties import (
    compute_net_charge,
    compute_molecular_weight,
    compute_gravy,
    compute_hydrophobic_moment,
    compute_isoelectric_point,
    compute_instability_index,
    compute_aromaticity,
    compute_aliphatic_index,
    compute_boman_index,
    compute_aa_composition,
    compute_all_properties,
    AA_MW,
)


# --- Net Charge ---

@pytest.mark.unit
class TestNetCharge:
    def test_polyK_positive(self):
        """Poly-lysine at pH 7 should be ~+10 (each K pKa=10.5, fully protonated)."""
        charge = compute_net_charge("KKKKKKKKKK", 7.0)
        assert charge > 9.5

    def test_polyD_negative(self):
        """Poly-aspartate at pH 7 should be ~-10 (each D pKa=3.9, fully deprotonated)."""
        charge = compute_net_charge("DDDDDDDDDD", 7.0)
        assert charge < -9.5

    def test_neutral_glycine(self):
        """Poly-glycine has only terminal charges, near zero at pH 7."""
        charge = compute_net_charge("GGGGGGGGGG", 7.0)
        assert abs(charge) < 0.5

    def test_ph_extremes(self):
        """At pH 0 charge is maximally positive; at pH 14 maximally negative."""
        seq = "KRKDEKR"
        charge_low = compute_net_charge(seq, 0.0)
        charge_high = compute_net_charge(seq, 14.0)
        assert charge_low > charge_high
        assert charge_low > 0
        assert charge_high < 0


# --- GRAVY ---

@pytest.mark.unit
class TestGRAVY:
    def test_hydrophobic_positive(self):
        """All hydrophobic residues → high positive GRAVY."""
        assert compute_gravy("ILVILVILV") > 3.0

    def test_hydrophilic_negative(self):
        """Charged/polar residues → negative GRAVY."""
        assert compute_gravy("RKDERKDER") < -3.0

    def test_empty_returns_zero(self):
        assert compute_gravy("") == 0.0


# --- Molecular Weight ---

@pytest.mark.unit
class TestMolecularWeight:
    def test_glycine_tripeptide(self):
        """GGG = 3×75.03 − 2×18.02 = 189.05 Da."""
        expected = 3 * 75.03 - 2 * 18.02
        result = compute_molecular_weight("GGG")
        assert abs(result - expected) < 0.1

    def test_monotonic_with_length(self):
        """Adding residues always increases MW."""
        mw_5 = compute_molecular_weight("AAAAA")
        mw_10 = compute_molecular_weight("AAAAAAAAAA")
        assert mw_10 > mw_5


# --- Hydrophobic Moment ---

@pytest.mark.unit
class TestHydrophobicMoment:
    def test_short_returns_zero(self):
        """Sequences < 3 aa return 0.0."""
        assert compute_hydrophobic_moment("AA") == 0.0

    def test_amphipathic_magainin(self, known_peptides):
        """Magainin 2 is a known amphipathic helix, HM > 0.3."""
        hm = compute_hydrophobic_moment(known_peptides["Magainin 2"])
        assert hm > 0.3

    def test_non_negative(self):
        """HM should always be >= 0 (it's a magnitude)."""
        assert compute_hydrophobic_moment("AAAKKK") >= 0


# --- Isoelectric Point ---

@pytest.mark.unit
class TestIsoelectricPoint:
    def test_basic_peptide(self):
        """Poly-lysine pI should be > 10."""
        assert compute_isoelectric_point("KKKKKK") > 10

    def test_acidic_peptide(self):
        """Poly-aspartate pI should be < 4."""
        assert compute_isoelectric_point("DDDDDD") < 4

    def test_in_valid_range(self):
        """pI must always be between 0 and 14."""
        pi = compute_isoelectric_point("GIGKFLHSAKKFGKAFVGEIMNS")
        assert 0 < pi < 14


# --- Instability Index ---

@pytest.mark.unit
class TestInstabilityIndex:
    def test_stable_poly_ala(self):
        """Poly-alanine has no unstable dipeptides → 0.0."""
        assert compute_instability_index("AAAAAAA") == 0.0

    def test_unstable_dipeptides(self):
        """DG is in the unstable list → index > 0."""
        assert compute_instability_index("DGDGDG") > 0

    def test_single_residue(self):
        """Single residue has no dipeptides → 0.0."""
        assert compute_instability_index("A") == 0.0


# --- Aromaticity ---

@pytest.mark.unit
class TestAromaticity:
    def test_all_aromatic(self):
        """100% aromatic residues (F, W, Y)."""
        assert compute_aromaticity("FWY") == 1.0

    def test_no_aromatic(self):
        """No aromatic residues."""
        assert compute_aromaticity("AAAA") == 0.0

    def test_fraction(self):
        """One aromatic in 4 residues → 0.25."""
        assert abs(compute_aromaticity("FAAA") - 0.25) < 0.01


# --- Aliphatic Index ---

@pytest.mark.unit
class TestAliphaticIndex:
    def test_empty(self):
        assert compute_aliphatic_index("") == 0.0

    def test_all_alanine(self):
        """100% A → aliphatic index = 100."""
        assert abs(compute_aliphatic_index("AAAAAAAAAA") - 100.0) < 0.1


# --- Boman Index ---

@pytest.mark.unit
class TestBomanIndex:
    def test_empty(self):
        assert compute_boman_index("") == 0.0

    def test_arginine_high(self):
        """Arginine has very high Boman value (14.92)."""
        boman = compute_boman_index("RRRR")
        assert boman > 10


# --- compute_all_properties ---

@pytest.mark.unit
class TestAllProperties:
    def test_short_returns_none(self):
        """Sequences < 3 AA after cleaning return None."""
        assert compute_all_properties("AB") is None
        assert compute_all_properties("KK") is None

    def test_full_structure(self, known_peptides):
        """Magainin 2 result has 31 keys (11 props + 20 AA compositions)."""
        result = compute_all_properties(known_peptides["Magainin 2"])
        assert result is not None
        # 11 named properties + "sequence" + "length" + 20 aa_ keys = 33
        # Actually: sequence, length, molecular_weight, net_charge_ph7,
        # isoelectric_point, gravy, hydrophobic_moment, instability_index,
        # aromaticity, aliphatic_index, boman_index = 11 keys + 20 aa_ = 31
        expected_keys = {
            "sequence", "length", "molecular_weight", "net_charge_ph7",
            "isoelectric_point", "gravy", "hydrophobic_moment",
            "instability_index", "aromaticity", "aliphatic_index", "boman_index",
        }
        aa_keys = {f"aa_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"}
        all_expected = expected_keys | aa_keys
        assert set(result.keys()) == all_expected

    def test_all_values_finite(self, known_peptides):
        """No NaN or Inf in any computed property."""
        for name, seq in known_peptides.items():
            result = compute_all_properties(seq)
            assert result is not None, f"Failed for {name}"
            for key, val in result.items():
                if isinstance(val, (int, float)):
                    assert math.isfinite(val), f"{name}.{key} = {val}"
