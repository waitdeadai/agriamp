"""AgriAMP Agent — Orchestrates 6 bioinformatics tools in an agentic workflow."""

import time
import random
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from tools import ToolResult
from tools.data_query import DataQueryTool
from tools.embeddings import EmbeddingsTool
from tools.properties import compute_all_properties
from tools.classifier import ClassifierTool
from tools.generator import GeneratorTool
from tools.toxicity import ToxicityTool

# Path to pre-computed embeddings (run precompute_embeddings.py to generate)
CACHED_EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "modlamp_embeddings.npz"
)


@dataclass
class AgentStep:
    tool_name: str
    icon: str
    status: str  # "running" | "success" | "warning" | "error"
    message: str
    duration: float = 0.0
    data: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    steps: list[AgentStep] = field(default_factory=list)
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)
    total_duration: float = 0.0
    success: bool = False


class AgriAMPAgent:
    """Agentic pipeline for antimicrobial peptide discovery."""

    def __init__(self):
        self.data_tool = DataQueryTool()
        self.embedding_tool = EmbeddingsTool()
        self.classifier_tool = ClassifierTool()
        self.generator_tool = GeneratorTool()
        self.toxicity_tool = ToxicityTool()

    def _update_step(self, result, step_result, callback):
        """Update the last step with tool result."""
        result.steps[-1].status = step_result.status
        result.steps[-1].message = step_result.message
        result.steps[-1].duration = step_result.duration
        if callback:
            callback(result.steps[-1])

    def run(self, pathogen: str = "Botrytis cinerea", callback=None) -> AgentResult:
        """Run the full 6-step agentic pipeline."""
        result = AgentResult()
        start_total = time.time()
        random.seed(42)

        def notify(step: AgentStep):
            result.steps.append(step)
            if callback:
                callback(step)

        # ═══════════════════════════════════════════════════════════
        # STEP 1: Query AMP databases
        # ═══════════════════════════════════════════════════════════
        notify(AgentStep("Database Query", "🔬", "running",
                         "Querying antimicrobial peptide databases..."))
        r1 = self.data_tool.run(pathogen=pathogen)
        self._update_step(result, r1, callback)

        if r1.status == "error":
            result.total_duration = time.time() - start_total
            return result

        seed_sequences = r1.data["seed_sequences"]
        all_amp_seqs = r1.data["amp_training"]
        non_amp_seqs = r1.data["non_amp_training"]

        # ═══════════════════════════════════════════════════════════
        # STEP 2: Generate variants from seed peptides
        # ═══════════════════════════════════════════════════════════
        notify(AgentStep("Variant Generator", "🧪", "running",
                         f"Generating optimized variants from {len(seed_sequences)} seed peptides..."))
        r2 = self.generator_tool.run(seed_sequences=seed_sequences, max_total=150)
        self._update_step(result, r2, callback)

        variant_sequences = r2.data.get("variant_sequences", [])
        all_candidate_seqs = list(seed_sequences) + list(variant_sequences)

        # ═══════════════════════════════════════════════════════════
        # STEP 3: ESM-2 Embeddings (cached training + live candidates)
        # ═══════════════════════════════════════════════════════════
        def clean_seq(s):
            return "".join(c for c in s.upper() if c in "ACDEFGHIKLMNPQRSTVWY")

        use_cache = os.path.exists(CACHED_EMBEDDINGS_PATH)

        if use_cache:
            # Load pre-computed embeddings for training data
            cached = np.load(CACHED_EMBEDDINGS_PATH, allow_pickle=True)
            cached_seqs = cached["sequences"].tolist()
            cached_labels = cached["labels"].tolist()
            cached_embs = cached["embeddings"]
            embed_dim = cached_embs.shape[1]

            n_cached = len(cached_seqs)
            n_amp_cached = sum(cached_labels)
            n_non_cached = n_cached - n_amp_cached

            notify(AgentStep("ESM-2 Embeddings", "🧬", "running",
                             f"Loading {n_cached} pre-computed embeddings ({n_amp_cached} AMPs + {n_non_cached} non-AMPs) "
                             f"+ generating embeddings for {len(all_candidate_seqs)} candidates with ESM-2 (650M params)..."))

            # Build cached lookup
            cached_seq_to_emb = {}
            cached_seq_to_label = {}
            for i, seq in enumerate(cached_seqs):
                if seq not in cached_seq_to_emb:
                    cached_seq_to_emb[seq] = cached_embs[i]
                    cached_seq_to_label[seq] = cached_labels[i]

            # Map training sequences to cached embeddings
            sampled_amps = []
            sampled_non = []
            for seq in all_amp_seqs:
                cs = clean_seq(seq)
                if cs in cached_seq_to_emb:
                    sampled_amps.append(cs)
            for seq in non_amp_seqs:
                cs = clean_seq(seq)
                if cs in cached_seq_to_emb:
                    sampled_non.append(cs)

            # Embed only candidates (fast: ~160 sequences)
            r3 = self.embedding_tool.run(sequences=all_candidate_seqs)
            self._update_step(result, r3, callback)

            if r3.status == "error":
                result.total_duration = time.time() - start_total
                return result

            cand_embeddings = r3.data["embeddings"]
            cand_clean_seqs = r3.data["sequences"]

            # Unified lookup: cached + candidate
            seq_to_emb = dict(cached_seq_to_emb)
            for i, seq in enumerate(cand_clean_seqs):
                if seq not in seq_to_emb:
                    seq_to_emb[seq] = cand_embeddings[i]

        else:
            # No cache — embed everything (original behavior)
            sampled_amps = list(all_amp_seqs)
            sampled_non = list(non_amp_seqs)
            embed_dim = 1280  # default ESM-2 650M

            total_to_embed = len(sampled_amps) + len(sampled_non) + len(all_candidate_seqs)
            notify(AgentStep("ESM-2 Embeddings", "🧬", "running",
                             f"Generating embeddings with ESM-2 (650M params) for {total_to_embed} sequences "
                             f"({len(sampled_amps)} AMPs + {len(sampled_non)} non-AMPs + {len(all_candidate_seqs)} candidates)..."))

            all_seqs = list(sampled_amps) + list(sampled_non) + list(all_candidate_seqs)
            r3 = self.embedding_tool.run(sequences=all_seqs)
            self._update_step(result, r3, callback)

            if r3.status == "error":
                result.total_duration = time.time() - start_total
                return result

            embeddings_all = r3.data["embeddings"]
            clean_seqs_all = r3.data["sequences"]
            embed_dim = embeddings_all.shape[1]

            seq_to_emb = {}
            for i, seq in enumerate(clean_seqs_all):
                if seq not in seq_to_emb:
                    seq_to_emb[seq] = embeddings_all[i]

        def get_aligned(seqs):
            """Get embeddings + properties aligned by valid sequences."""
            embs, props, valid = [], [], []
            for s in seqs:
                cs = clean_seq(s)
                if cs in seq_to_emb:
                    p = compute_all_properties(cs)
                    if p is not None:
                        embs.append(seq_to_emb[cs])
                        props.append(p)
                        valid.append(cs)
            return (np.array(embs) if embs else np.empty((0, embed_dim)),
                    props, valid)

        # ═══════════════════════════════════════════════════════════
        # STEP 4: Biochemical Properties
        # ═══════════════════════════════════════════════════════════
        t4_start = time.time()
        notify(AgentStep("Biochemical Properties", "📊", "running",
                         "Computing 12 biochemical properties (charge, amphipathicity, GRAVY, MW, pI, Boman...)..."))

        amp_embs, amp_props, valid_amp = get_aligned(sampled_amps)
        non_embs, non_props, valid_non = get_aligned(sampled_non)
        cand_embs, cand_props, valid_cand = get_aligned(all_candidate_seqs)

        # Stats for message
        n_ideal_charge = sum(1 for p in cand_props if 2 <= p["net_charge_ph7"] <= 9)
        n_ideal_hm = sum(1 for p in cand_props if p["hydrophobic_moment"] > 0.3)
        avg_charge = np.mean([p["net_charge_ph7"] for p in cand_props]) if cand_props else 0

        t4_dur = time.time() - t4_start
        result.steps[-1].status = "success"
        result.steps[-1].message = (
            f"Computed 12 biochemical properties for {len(cand_props)} candidates. "
            f"Effective AMPs have charge +2 to +9 and high hydrophobic moment. "
            f"Of {len(cand_props)} candidates: {n_ideal_charge} with ideal charge, "
            f"{n_ideal_hm} with high amphipathicity. Average charge: {avg_charge:+.1f}."
        )
        result.steps[-1].duration = t4_dur
        if callback:
            callback(result.steps[-1])

        # ═══════════════════════════════════════════════════════════
        # STEP 5: AMP Classifier (Random Forest)
        # ═══════════════════════════════════════════════════════════
        notify(AgentStep("AMP Classifier", "🤖", "running",
                         f"Training ML classifier ({len(amp_props)} AMPs + {len(non_props)} non-AMPs)..."))

        if len(amp_props) >= 10 and len(non_props) >= 10:
            r5 = self.classifier_tool.run(
                amp_embeddings=amp_embs,
                amp_properties=amp_props,
                non_amp_embeddings=non_embs,
                non_amp_properties=non_props,
                candidate_embeddings=cand_embs,
                candidate_properties=cand_props,
            )
            self._update_step(result, r5, callback)
            amp_probs = r5.data.get("amp_probabilities", [0.5] * len(cand_props))
            result.metrics = r5.data.get("metrics", {})
            result.metrics["feature_importances"] = r5.data.get("feature_importances", {})
            result.metrics["top_features"] = r5.data.get("top_features", [])
        else:
            # Property-based scoring fallback
            amp_probs = []
            for p in cand_props:
                charge_s = min(max(p["net_charge_ph7"] / 6, 0), 1)
                hm_s = min(max(p["hydrophobic_moment"] / 0.8, 0), 1)
                amp_probs.append(0.5 * charge_s + 0.5 * hm_s)
            result.steps[-1].status = "warning"
            result.steps[-1].message = (
                f"Insufficient data for ML. Used property-based scoring "
                f"(charge + hydrophobic moment) for {len(cand_props)} candidates."
            )
            result.metrics = {"cv_auc_mean": None, "method": "property_scoring"}
            if callback:
                callback(result.steps[-1])

        # ═══════════════════════════════════════════════════════════
        # STEP 6: Toxicity Screening
        # ═══════════════════════════════════════════════════════════
        notify(AgentStep("Toxicity Screening", "🛡️", "running",
                         f"Evaluating toxicity risk for {len(cand_props)} candidates..."))
        cand_seqs = [p["sequence"] for p in cand_props]
        r6 = self.toxicity_tool.run(sequences=cand_seqs, properties_list=cand_props)
        self._update_step(result, r6, callback)

        tox_results = r6.data.get("toxicity_results", [])

        # ═══════════════════════════════════════════════════════════
        # BUILD FINAL CANDIDATES TABLE
        # ═══════════════════════════════════════════════════════════
        curated_lookup = {}
        for amp_info in r1.data.get("curated_amps", []):
            cs = clean_seq(amp_info["sequence"])
            curated_lookup[cs] = amp_info["name"]

        seed_set = {clean_seq(s) for s in seed_sequences}

        candidates = []
        for i, props in enumerate(cand_props):
            seq = props["sequence"]
            amp_prob = amp_probs[i] if i < len(amp_probs) else 0.5
            tox = tox_results[i] if i < len(tox_results) else {
                "toxicity_risk": 0.5, "selectivity_score": 0.5,
                "passed_screening": False, "flags": []
            }

            # Composite AgriAMP Score
            charge_norm = min(max(props["net_charge_ph7"] / 6, 0), 1)
            hm_norm = min(max(props["hydrophobic_moment"] / 0.8, 0), 1)
            stability_norm = max(1 - props["instability_index"] / 100, 0)
            tox_risk = tox.get("toxicity_risk", 0.5)

            agriamp_score = (
                0.35 * amp_prob
                + 0.25 * charge_norm
                + 0.20 * hm_norm
                + 0.10 * stability_norm
                + 0.10 * (1 - tox_risk)
            )

            candidates.append({
                "sequence": seq,
                "length": len(seq),
                "agriamp_score": round(agriamp_score, 4),
                "amp_probability": round(amp_prob, 4),
                "net_charge": round(props["net_charge_ph7"], 2),
                "molecular_weight": round(props["molecular_weight"], 1),
                "gravy": round(props["gravy"], 3),
                "hydrophobic_moment": round(props["hydrophobic_moment"], 4),
                "isoelectric_point": round(props["isoelectric_point"], 2),
                "instability_index": round(props["instability_index"], 1),
                "aromaticity": round(props["aromaticity"], 3),
                "aliphatic_index": round(props["aliphatic_index"], 1),
                "boman_index": round(props["boman_index"], 3),
                "toxicity_risk": round(tox_risk, 3),
                "selectivity_score": tox.get("selectivity_score", 0.5),
                "passed_toxicity": tox.get("passed_screening", False),
                "tox_flags": tox.get("flags", []),
                "origin": "seed" if seq in seed_set else "variant",
                "name": curated_lookup.get(seq, ""),
            })

        df = pd.DataFrame(candidates)
        df = df.sort_values("agriamp_score", ascending=False).reset_index(drop=True)
        result.candidates = df
        result.total_duration = time.time() - start_total
        result.success = True

        return result


if __name__ == "__main__":
    """CLI test of the full pipeline."""
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("AgriAMP -- Full Pipeline Test")
    print("=" * 60)

    agent = AgriAMPAgent()

    def print_step(step):
        icons = {"success": "[OK]", "warning": "[WARN]", "error": "[ERR]", "running": "[...]"}
        print(f"\n[{step.tool_name}] {icons.get(step.status, '?')}")
        # Strip emojis for console
        msg = step.message.encode('ascii', 'ignore').decode('ascii')
        print(f"  {msg}")
        if step.duration > 0:
            print(f"  Duration: {step.duration:.1f}s")

    result = agent.run(pathogen="Botrytis cinerea", callback=print_step)

    print(f"\n{'=' * 60}")
    print(f"Pipeline {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Total duration: {result.total_duration:.1f}s")
    print(f"Candidates: {len(result.candidates)}")

    if result.success and len(result.candidates) > 0:
        print(f"\nTop 5 candidates:")
        top5 = result.candidates.head(5)
        for _, row in top5.iterrows():
            name = row["name"] or row["sequence"][:15] + "..."
            print(f"  {name}: Score={row['agriamp_score']:.4f} "
                  f"Charge={row['net_charge']:+.1f} "
                  f"Tox={row['toxicity_risk']:.2f} "
                  f"{'✅' if row['passed_toxicity'] else '⚠️'}")

        if result.metrics.get("cv_auc_mean"):
            print(f"\nClassifier AUC: {result.metrics['cv_auc_mean']:.3f}")
