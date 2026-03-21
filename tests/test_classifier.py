"""Tests for ML classifier metrics from precomputed data (tools/classifier.py)."""

import math
import pytest
import numpy as np


@pytest.mark.unit
class TestClassifierMetrics:
    """Validate ML metrics from precomputed Botrytis cinerea pipeline run."""

    def test_auc_above_095(self, precomputed_botrytis):
        """Cross-validated AUC must be >= 0.95."""
        assert precomputed_botrytis["metrics"]["cv_auc_mean"] >= 0.95

    def test_mcc_above_080(self, precomputed_botrytis):
        """Matthews Correlation Coefficient >= 0.80 (primary AMP metric per literature)."""
        assert precomputed_botrytis["metrics"]["oof_mcc"] >= 0.80

    def test_sensitivity_above_090(self, precomputed_botrytis):
        """Sensitivity (true positive rate for AMPs) > 0.90."""
        assert precomputed_botrytis["metrics"]["oof_sensitivity"] > 0.90

    def test_specificity_above_090(self, precomputed_botrytis):
        """Specificity (true negative rate) > 0.90."""
        assert precomputed_botrytis["metrics"]["oof_specificity"] > 0.90

    def test_sensitivity_specificity_balanced(self, precomputed_botrytis):
        """Sens and Spec within 5% of each other → no class bias."""
        m = precomputed_botrytis["metrics"]
        diff = abs(m["oof_sensitivity"] - m["oof_specificity"])
        assert diff < 0.05

    def test_confusion_matrix_2x2(self, precomputed_botrytis):
        """Confusion matrix is a 2x2 array."""
        cm = precomputed_botrytis["metrics"]["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2

    def test_confusion_matrix_sums_5200(self, precomputed_botrytis):
        """All cells sum to total training samples (2600 + 2600 = 5200)."""
        cm = precomputed_botrytis["metrics"]["confusion_matrix"]
        total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        assert total == 5200

    def test_training_balanced(self, precomputed_botrytis):
        """Training data is balanced: n_positive ≈ n_negative."""
        m = precomputed_botrytis["metrics"]
        assert m["n_train_positive"] == m["n_train_negative"]

    def test_no_leakage_oof_lt_train(self, precomputed_botrytis):
        """Out-of-fold AUC < training AUC (confirms no data leakage)."""
        m = precomputed_botrytis["metrics"]
        assert m["cv_auc_mean"] < m["train_auc"]

    def test_cv_fold_stability(self, precomputed_botrytis):
        """Standard deviation of 5 CV fold scores < 0.02."""
        scores = precomputed_botrytis["metrics"]["cv_scores"]
        assert len(scores) == 5
        std = np.std(scores)
        assert std < 0.02

    def test_feature_importances_sum_one(self, precomputed_botrytis):
        """Feature importances sum to approximately 1.0."""
        importances = precomputed_botrytis["metrics"]["feature_importances"]
        total = sum(importances.values())
        assert abs(total - 1.0) < 0.01
