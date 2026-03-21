"""Unit tests for variant generator (tools/generator.py)."""

import random
import pytest
from tools.generator import GeneratorTool, CATIONIC_AA, HYDROPHOBIC_AA, NEUTRAL_AA, ANIONIC_AA


SEED_PEPTIDE = "GIGKFLHSAKKFGKAFVGEIMNS"  # Magainin 2, 23 aa
ENHANCING_AAS = set(CATIONIC_AA + HYDROPHOBIC_AA)  # {K, R, L, W, F, I}


@pytest.fixture
def generator():
    return GeneratorTool()


@pytest.mark.unit
class TestReproducibility:
    def test_seed42_deterministic(self, generator):
        """Two runs with same seed produce identical output."""
        r1 = generator._execute(seed_sequences=[SEED_PEPTIDE], max_total=50)
        r2 = generator._execute(seed_sequences=[SEED_PEPTIDE], max_total=50)
        seqs1 = r1.data["variant_sequences"]
        seqs2 = r2.data["variant_sequences"]
        assert seqs1 == seqs2


@pytest.mark.unit
class TestPointMutations:
    def test_single_position_change(self, generator):
        """Each point mutation differs from parent at exactly 1 position."""
        result = generator._execute(seed_sequences=[SEED_PEPTIDE], max_total=200)
        point_muts = [v for v in result.data["variants"] if v["strategy"] == "point_mutation"]
        for v in point_muts:
            parent = v["parent"]
            variant = v["sequence"]
            assert len(variant) == len(parent), "Point mutation should not change length"
            diffs = sum(1 for a, b in zip(parent, variant) if a != b)
            assert diffs == 1, f"Expected 1 diff, got {diffs}: {v['description']}"

    def test_uses_enhancing_aas(self, generator):
        """Substituted amino acids are in the enhancing set (K/R/L/W/F/I)."""
        result = generator._execute(seed_sequences=[SEED_PEPTIDE], max_total=200)
        point_muts = [v for v in result.data["variants"] if v["strategy"] == "point_mutation"]
        for v in point_muts:
            parent = v["parent"]
            variant = v["sequence"]
            for a, b in zip(parent, variant):
                if a != b:
                    assert b in ENHANCING_AAS, f"Unexpected substitution: {a}→{b}"


@pytest.mark.unit
class TestChargeOptimization:
    def test_targets_neutral_or_anionic(self, generator):
        """Charge optimization only mutates neutral or anionic residues."""
        result = generator._execute(seed_sequences=[SEED_PEPTIDE], max_total=200)
        charge_opts = [v for v in result.data["variants"] if v["strategy"] == "charge_optimization"]
        mutable = NEUTRAL_AA | ANIONIC_AA
        for v in charge_opts:
            parent = v["parent"]
            variant = v["sequence"]
            for i, (a, b) in enumerate(zip(parent, variant)):
                if a != b:
                    assert a in mutable, f"Mutated non-mutable {a} at pos {i}"


@pytest.mark.unit
class TestTruncations:
    def test_min_length_respected(self, generator):
        """No truncation produces a sequence shorter than 8 AA."""
        result = generator._execute(seed_sequences=[SEED_PEPTIDE], max_total=200)
        truncations = [v for v in result.data["variants"]
                       if "truncation" in v["strategy"]]
        for v in truncations:
            assert len(v["sequence"]) >= 8, f"Too short: {len(v['sequence'])} aa"

    def test_short_peptide_no_truncations(self, generator):
        """8-AA peptide cannot be truncated (would go below min_length)."""
        short = "KKLLKKLL"  # exactly 8 aa
        result = generator._execute(seed_sequences=[short], max_total=200)
        truncations = [v for v in result.data["variants"]
                       if "truncation" in v["strategy"]]
        assert len(truncations) == 0


@pytest.mark.unit
class TestScrambledControls:
    def test_same_composition(self, generator):
        """Scrambled controls have identical AA composition as parent."""
        result = generator._execute(seed_sequences=[SEED_PEPTIDE], max_total=200)
        scrambled = [v for v in result.data["variants"]
                     if v["strategy"] == "scrambled_control"]
        for v in scrambled:
            assert sorted(v["sequence"]) == sorted(v["parent"])

    def test_differs_from_parent(self, generator):
        """Scrambled sequence is not identical to parent."""
        result = generator._execute(seed_sequences=[SEED_PEPTIDE], max_total=200)
        scrambled = [v for v in result.data["variants"]
                     if v["strategy"] == "scrambled_control"]
        for v in scrambled:
            assert v["sequence"] != v["parent"]


@pytest.mark.unit
class TestGeneratorConstraints:
    def test_no_duplicates(self, generator):
        """All variant sequences are unique."""
        result = generator._execute(seed_sequences=[SEED_PEPTIDE], max_total=200)
        seqs = result.data["variant_sequences"]
        assert len(seqs) == len(set(seqs))

    def test_max_total_respected(self, generator):
        """max_total caps the number of variants."""
        result = generator._execute(seed_sequences=[SEED_PEPTIDE], max_total=10)
        assert len(result.data["variant_sequences"]) <= 10

    def test_empty_seeds_error(self, generator):
        """Empty seed list returns error."""
        result = generator._execute(seed_sequences=[], max_total=50)
        assert result.status == "error"

    def test_all_strategies_present(self, generator):
        """With sufficient seeds and limit, all 5 strategies appear."""
        result = generator._execute(seed_sequences=[SEED_PEPTIDE], max_total=200)
        strategies = {v["strategy"] for v in result.data["variants"]}
        expected = {"point_mutation", "charge_optimization",
                    "n_terminal_truncation", "c_terminal_truncation",
                    "scrambled_control"}
        assert strategies == expected
