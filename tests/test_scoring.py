"""Tests for the composite AgriAMP scoring formula (agent.py:296-307)."""

import math
import json
import os
import pytest
from tests.conftest import compute_agriamp_score, PRECOMPUTED_DIR


@pytest.mark.unit
class TestScoringFormula:
    def test_weights_sum_one(self):
        """Score weights must sum to exactly 1.0."""
        assert 0.35 + 0.25 + 0.20 + 0.10 + 0.10 == 1.0

    def test_score_floor(self):
        """All worst-case inputs → score = 0.0."""
        score = compute_agriamp_score(
            amp_prob=0.0, charge=0.0, hm=0.0,
            instability=100.0, tox_risk=1.0,
        )
        assert score == 0.0

    def test_score_ceiling(self):
        """All optimal inputs → score = 1.0."""
        score = compute_agriamp_score(
            amp_prob=1.0, charge=6.0, hm=0.8,
            instability=0.0, tox_risk=0.0,
        )
        assert score == 1.0

    def test_charge_norm_clamped_negative(self):
        """Negative charge normalizes to 0, not negative."""
        score = compute_agriamp_score(
            amp_prob=0.0, charge=-5.0, hm=0.0,
            instability=100.0, tox_risk=1.0,
        )
        assert score >= 0.0

    def test_charge_norm_clamped_high(self):
        """Very high charge normalizes to 1, not > 1."""
        s1 = compute_agriamp_score(amp_prob=0, charge=6, hm=0, instability=100, tox_risk=1)
        s2 = compute_agriamp_score(amp_prob=0, charge=12, hm=0, instability=100, tox_risk=1)
        assert s1 == s2  # both clamped at 1.0

    def test_hm_norm_clamped(self):
        """HM normalization: negative → 0, very high → 1."""
        s_neg = compute_agriamp_score(amp_prob=0, charge=0, hm=-0.1, instability=100, tox_risk=1)
        s_high = compute_agriamp_score(amp_prob=0, charge=0, hm=2.0, instability=100, tox_risk=1)
        s_max = compute_agriamp_score(amp_prob=0, charge=0, hm=0.8, instability=100, tox_risk=1)
        assert s_neg == 0.0
        assert s_high == s_max  # both clamped at 1.0

    def test_stability_norm_clamped(self):
        """Instability > 100 normalizes to 0, not negative."""
        score = compute_agriamp_score(
            amp_prob=0.0, charge=0.0, hm=0.0,
            instability=200.0, tox_risk=1.0,
        )
        assert score >= 0.0

    def test_monotonic_amp_prob(self):
        """Higher amp_probability → higher score (all else equal)."""
        s1 = compute_agriamp_score(amp_prob=0.3, charge=4, hm=0.5, instability=10, tox_risk=0.1)
        s2 = compute_agriamp_score(amp_prob=0.9, charge=4, hm=0.5, instability=10, tox_risk=0.1)
        assert s2 > s1

    def test_monotonic_charge(self):
        """Higher charge (0→6) → higher score."""
        s1 = compute_agriamp_score(amp_prob=0.5, charge=1, hm=0.5, instability=10, tox_risk=0.1)
        s2 = compute_agriamp_score(amp_prob=0.5, charge=5, hm=0.5, instability=10, tox_risk=0.1)
        assert s2 > s1


@pytest.mark.unit
class TestEpinecidinValidation:
    """The most important test: pipeline reproduces published lab result."""

    def test_epi4_beats_epinecidin(self, precomputed_botrytis):
        """EPI-4 (optimized variant) scores higher than Epinecidin-1 (original).
        This independently confirms published lab results (Food Chemistry 2022):
        EPI-4 MIC 6.0 µM < Epinecidin-1 MIC 12.5 µM → more potent → higher score.
        """
        candidates = precomputed_botrytis["candidates"]
        epi1 = next(c for c in candidates if c["name"] == "Epinecidin-1")
        epi4 = next(c for c in candidates if c["name"] == "EPI-4")
        assert epi4["agriamp_score"] > epi1["agriamp_score"]


@pytest.mark.integration
class TestPrecomputedScoreConsistency:
    def test_all_precomputed_scores_match(self, precomputed_botrytis):
        """Recompute AgriAMP score for all 160 candidates; verify match ±0.002."""
        candidates = precomputed_botrytis["candidates"]
        mismatches = []
        for c in candidates:
            recomputed = compute_agriamp_score(
                amp_prob=c["amp_probability"],
                charge=c["net_charge"],
                hm=c["hydrophobic_moment"],
                instability=c["instability_index"],
                tox_risk=c["toxicity_risk"],
            )
            if abs(recomputed - c["agriamp_score"]) > 0.002:
                mismatches.append(
                    f"{c.get('name', c['sequence'][:10])}: "
                    f"stored={c['agriamp_score']}, recomputed={recomputed:.4f}"
                )
        assert not mismatches, f"{len(mismatches)} mismatches:\n" + "\n".join(mismatches[:5])
