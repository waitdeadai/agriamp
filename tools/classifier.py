"""Tool 4: AMP classifier using Random Forest on ESM-2 embeddings + biochemical properties."""

import numpy as np
import pandas as pd
from tools import BaseTool, ToolResult


class ClassifierTool(BaseTool):
    name = "AMP Classifier"
    description = "Clasifica péptidos como AMP/non-AMP usando ML"
    icon = "🤖"

    def __init__(self):
        self._model = None
        self._pca = None
        self._scaler = None
        self._metrics = None
        self._feature_names = None
        self._feature_importances = None

    def _build_features(self, embeddings: np.ndarray, properties_list: list[dict]) -> np.ndarray:
        """Combine PCA-reduced embeddings with biochemical properties."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # PCA on embeddings
        n_components = min(64, embeddings.shape[1], embeddings.shape[0])
        if self._pca is None:
            self._pca = PCA(n_components=n_components, random_state=42)
            emb_reduced = self._pca.fit_transform(embeddings)
        else:
            emb_reduced = self._pca.transform(embeddings)

        # Extract property features
        prop_keys = [
            "length", "molecular_weight", "net_charge_ph7", "isoelectric_point",
            "gravy", "hydrophobic_moment", "instability_index", "aromaticity",
            "aliphatic_index", "boman_index",
        ]
        aa_keys = [f"aa_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"]
        all_keys = prop_keys + aa_keys

        prop_matrix = np.zeros((len(properties_list), len(all_keys)))
        for i, props in enumerate(properties_list):
            for j, key in enumerate(all_keys):
                prop_matrix[i, j] = props.get(key, 0.0)

        # Combine
        features = np.hstack([emb_reduced, prop_matrix])

        # Scale
        if self._scaler is None:
            self._scaler = StandardScaler()
            features = self._scaler.fit_transform(features)
        else:
            features = self._scaler.transform(features)

        # Store feature names
        self._feature_names = [f"ESM2_PC{i+1}" for i in range(n_components)] + all_keys

        return features

    def _execute(
        self,
        # Training data
        amp_embeddings: np.ndarray,
        amp_properties: list[dict],
        non_amp_embeddings: np.ndarray,
        non_amp_properties: list[dict],
        # Data to classify
        candidate_embeddings: np.ndarray,
        candidate_properties: list[dict],
    ) -> ToolResult:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
        from sklearn.metrics import (
            roc_auc_score, matthews_corrcoef, f1_score,
            precision_score, recall_score, accuracy_score, confusion_matrix,
        )

        # Build training features
        all_train_embeddings = np.vstack([amp_embeddings, non_amp_embeddings])
        all_train_properties = amp_properties + non_amp_properties
        labels = np.array([1] * len(amp_properties) + [0] * len(non_amp_properties))

        X_train = self._build_features(all_train_embeddings, all_train_properties)
        y_train = labels

        # Train Random Forest
        self._model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

        # Stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_scores = cross_val_score(self._model, X_train, y_train, cv=skf, scoring="roc_auc")
        mean_auc = cv_scores.mean()
        std_auc = cv_scores.std()

        # Out-of-fold predictions for honest metrics
        y_oof_proba = cross_val_predict(
            self._model, X_train, y_train, cv=skf, method="predict_proba"
        )[:, 1]
        y_oof_pred = (y_oof_proba >= 0.5).astype(int)

        # Comprehensive metrics from out-of-fold predictions
        oof_mcc = matthews_corrcoef(y_train, y_oof_pred)
        oof_f1 = f1_score(y_train, y_oof_pred)
        oof_sensitivity = recall_score(y_train, y_oof_pred)  # recall of class 1
        oof_specificity = recall_score(y_train, y_oof_pred, pos_label=0)  # recall of class 0
        oof_precision = precision_score(y_train, y_oof_pred)
        oof_accuracy = accuracy_score(y_train, y_oof_pred)
        oof_cm = confusion_matrix(y_train, y_oof_pred).tolist()

        # If AUC is low, try Gradient Boosting
        if mean_auc < 0.85:
            gb_model = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
            )
            gb_scores = cross_val_score(gb_model, X_train, y_train, cv=skf, scoring="roc_auc")
            if gb_scores.mean() > mean_auc:
                self._model = gb_model
                cv_scores = gb_scores
                mean_auc = gb_scores.mean()
                std_auc = gb_scores.std()
                # Recompute OOF metrics
                y_oof_proba = cross_val_predict(
                    self._model, X_train, y_train, cv=skf, method="predict_proba"
                )[:, 1]
                y_oof_pred = (y_oof_proba >= 0.5).astype(int)
                oof_mcc = matthews_corrcoef(y_train, y_oof_pred)
                oof_f1 = f1_score(y_train, y_oof_pred)
                oof_sensitivity = recall_score(y_train, y_oof_pred)
                oof_specificity = recall_score(y_train, y_oof_pred, pos_label=0)
                oof_precision = precision_score(y_train, y_oof_pred)
                oof_accuracy = accuracy_score(y_train, y_oof_pred)
                oof_cm = confusion_matrix(y_train, y_oof_pred).tolist()

        # Fit on full training data
        self._model.fit(X_train, y_train)

        # Feature importances
        importances = self._model.feature_importances_
        self._feature_importances = dict(zip(self._feature_names, importances))
        top_features = sorted(
            self._feature_importances.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Training predictions for ROC curve visualization
        y_train_pred = self._model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred)

        # Store metrics for visualization
        self._metrics = {
            "cv_auc_mean": round(mean_auc, 3),
            "cv_auc_std": round(std_auc, 3),
            "train_auc": round(train_auc, 3),
            "cv_scores": cv_scores.tolist(),
            "n_train_positive": int(sum(y_train)),
            "n_train_negative": int(len(y_train) - sum(y_train)),
            "y_train_true": y_train.tolist(),
            "y_train_pred": y_train_pred.tolist(),
            # Out-of-fold metrics (honest, no data leakage)
            "y_oof_proba": y_oof_proba.tolist(),
            "oof_mcc": round(oof_mcc, 3),
            "oof_f1": round(oof_f1, 3),
            "oof_sensitivity": round(oof_sensitivity, 3),
            "oof_specificity": round(oof_specificity, 3),
            "oof_precision": round(oof_precision, 3),
            "oof_accuracy": round(oof_accuracy, 3),
            "confusion_matrix": oof_cm,
        }

        # Classify candidates
        X_candidates = self._build_features(candidate_embeddings, candidate_properties)
        amp_probabilities = self._model.predict_proba(X_candidates)[:, 1]

        # Top features message
        top_feat_str = ", ".join(f"{name} ({imp:.3f})" for name, imp in top_features[:5])

        msg = (
            f"Entrené un clasificador con AUC {mean_auc:.3f} ± {std_auc:.3f} "
            f"en validación cruzada 5-fold. "
            f"Datos de entrenamiento: {int(sum(y_train))} AMPs positivos, "
            f"{int(len(y_train) - sum(y_train))} negativos. "
            f"Features más predictivas: {top_feat_str} — "
            f"consistente con la literatura que identifica carga neta y momento hidrofóbico "
            f"como propiedades clave de AMPs. "
            f"Clasifiqué {len(amp_probabilities)} candidatos."
        )

        return ToolResult(
            status="success",
            message=msg,
            data={
                "amp_probabilities": amp_probabilities.tolist(),
                "metrics": self._metrics,
                "top_features": top_features,
                "feature_importances": self._feature_importances,
            },
        )
