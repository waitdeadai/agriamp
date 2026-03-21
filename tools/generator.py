"""Tool 5: Generate peptide variants optimized for AMP activity."""

import random
from tools import BaseTool, ToolResult

# Amino acids that enhance AMP activity
CATIONIC_AA = ["K", "R"]  # Increase positive charge
HYDROPHOBIC_AA = ["L", "W", "F", "I"]  # Increase hydrophobicity
FLEXIBLE_AA = ["G", "P"]  # Structural flexibility

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")

# AAs grouped by property for smart substitutions
NEUTRAL_AA = set("AGSTNQ")
ANIONIC_AA = set("DE")


class GeneratorTool(BaseTool):
    name = "Variant Generator"
    description = "Generates optimized variants from seed peptides"
    icon = "🧪"

    def _generate_point_mutations(self, sequence: str, max_variants: int = 60) -> list[dict]:
        """Generate single-point substitution variants."""
        variants = []
        enhancing_aas = CATIONIC_AA + HYDROPHOBIC_AA

        for i, original_aa in enumerate(sequence):
            for new_aa in enhancing_aas:
                if new_aa != original_aa:
                    variant = sequence[:i] + new_aa + sequence[i + 1:]
                    variants.append({
                        "sequence": variant,
                        "strategy": "point_mutation",
                        "description": f"{original_aa}{i+1}{new_aa}",
                        "parent": sequence,
                    })
                    if len(variants) >= max_variants:
                        return variants
        return variants

    def _generate_charge_optimized(self, sequence: str, target_charge_range: tuple = (4, 6)) -> list[dict]:
        """Mutate neutral/anionic residues to increase positive charge."""
        variants = []
        mutable_positions = [
            i for i, aa in enumerate(sequence) if aa in NEUTRAL_AA or aa in ANIONIC_AA
        ]

        # Try replacing 1-3 positions
        for n_mutations in range(1, min(4, len(mutable_positions) + 1)):
            if len(mutable_positions) < n_mutations:
                break
            # Pick positions to mutate
            for _ in range(5):  # 5 random combinations per n_mutations
                positions = random.sample(mutable_positions, n_mutations)
                variant = list(sequence)
                for pos in positions:
                    variant[pos] = random.choice(CATIONIC_AA)
                variant_str = "".join(variant)
                if variant_str != sequence:
                    positions_str = "+".join(f"{sequence[p]}{p+1}{variant[p]}" for p in positions)
                    variants.append({
                        "sequence": variant_str,
                        "strategy": "charge_optimization",
                        "description": f"Charge+: {positions_str}",
                        "parent": sequence,
                    })

        return variants

    def _generate_truncations(self, sequence: str, min_length: int = 8) -> list[dict]:
        """Generate N- and C-terminal truncations."""
        variants = []
        seq_len = len(sequence)

        if seq_len <= min_length:
            return variants

        # N-terminal truncations (remove 1-5 residues from start)
        for n in range(1, min(6, seq_len - min_length + 1)):
            truncated = sequence[n:]
            variants.append({
                "sequence": truncated,
                "strategy": "n_terminal_truncation",
                "description": f"ΔN{n} ({seq_len}→{len(truncated)} aa)",
                "parent": sequence,
            })

        # C-terminal truncations
        for n in range(1, min(6, seq_len - min_length + 1)):
            truncated = sequence[:-n]
            variants.append({
                "sequence": truncated,
                "strategy": "c_terminal_truncation",
                "description": f"ΔC{n} ({seq_len}→{len(truncated)} aa)",
                "parent": sequence,
            })

        return variants

    def _generate_scrambled_controls(self, sequence: str, n_controls: int = 3) -> list[dict]:
        """Generate scrambled sequence controls (same composition, random order)."""
        variants = []
        aa_list = list(sequence)

        for i in range(n_controls):
            scrambled = aa_list.copy()
            random.shuffle(scrambled)
            scrambled_str = "".join(scrambled)
            if scrambled_str != sequence:
                variants.append({
                    "sequence": scrambled_str,
                    "strategy": "scrambled_control",
                    "description": f"Scramble #{i+1} (control negativo)",
                    "parent": sequence,
                })

        return variants

    def _execute(self, seed_sequences: list[str], max_total: int = 200) -> ToolResult:
        if not seed_sequences:
            return ToolResult(status="error", message="No hay secuencias semilla.")

        random.seed(42)  # Reproducibility

        all_variants = []
        seen = set()
        strategy_counts = {
            "point_mutation": 0,
            "charge_optimization": 0,
            "n_terminal_truncation": 0,
            "c_terminal_truncation": 0,
            "scrambled_control": 0,
        }

        variants_per_seed = max(10, max_total // len(seed_sequences))

        for seed in seed_sequences:
            if len(all_variants) >= max_total:
                break

            # Generate variants with each strategy
            mutations = self._generate_point_mutations(seed, max_variants=variants_per_seed // 2)
            charge_opt = self._generate_charge_optimized(seed)
            truncations = self._generate_truncations(seed)
            controls = self._generate_scrambled_controls(seed)

            for variant in mutations + charge_opt + truncations + controls:
                if variant["sequence"] not in seen and variant["sequence"] != seed:
                    seen.add(variant["sequence"])
                    all_variants.append(variant)
                    strategy_counts[variant["strategy"]] = strategy_counts.get(variant["strategy"], 0) + 1

                    if len(all_variants) >= max_total:
                        break

        # Build strategy summary
        strategy_summary = ", ".join(
            f"{count} {name.replace('_', ' ')}" for name, count in strategy_counts.items() if count > 0
        )

        msg = (
            f"Generated {len(all_variants)} variants from {len(seed_sequences)} seed peptides. "
            f"Strategies: {strategy_summary}. "
            f"Substitutions prioritize cationic residues (+K/R) to increase charge "
            f"and hydrophobic residues (+L/W/F) to improve amphipathicity. "
            f"Included scrambled controls to validate that sequence order matters."
        )

        return ToolResult(
            status="success",
            message=msg,
            data={
                "variants": all_variants,
                "variant_sequences": [v["sequence"] for v in all_variants],
                "n_variants": len(all_variants),
                "strategy_counts": strategy_counts,
                "n_seeds": len(seed_sequences),
            },
        )
