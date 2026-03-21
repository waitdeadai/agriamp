"""Tool 3: Calculate biochemical properties of peptide sequences."""

import numpy as np
import pandas as pd
from tools import BaseTool, ToolResult

# Kyte-Doolittle hydrophobicity scale
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Eisenberg scale for hydrophobic moment
EISENBERG_SCALE = {
    "A": 0.620, "R": -2.530, "N": -0.780, "D": -0.900, "C": 0.290,
    "Q": -0.850, "E": -0.740, "G": 0.480, "H": -0.400, "I": 1.380,
    "L": 1.060, "K": -1.500, "M": 0.640, "F": 1.190, "P": 0.120,
    "S": -0.180, "T": -0.050, "W": 0.810, "Y": 0.260, "V": 1.080,
}

# Amino acid molecular weights (Da)
AA_MW = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

# pKa values for charge calculation
POS_CHARGE_AA = {"K": 10.5, "R": 12.4, "H": 6.0}  # basic
NEG_CHARGE_AA = {"D": 3.9, "E": 4.1}  # acidic
N_TERM_PKA = 9.69
C_TERM_PKA = 2.34

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def compute_net_charge(sequence: str, ph: float = 7.0) -> float:
    """Compute net charge at given pH using Henderson-Hasselbalch."""
    charge = 0.0
    # N-terminus
    charge += 1.0 / (1.0 + 10 ** (ph - N_TERM_PKA))
    # C-terminus
    charge -= 1.0 / (1.0 + 10 ** (C_TERM_PKA - ph))
    # Side chains
    for aa in sequence:
        if aa in POS_CHARGE_AA:
            charge += 1.0 / (1.0 + 10 ** (ph - POS_CHARGE_AA[aa]))
        elif aa in NEG_CHARGE_AA:
            charge -= 1.0 / (1.0 + 10 ** (NEG_CHARGE_AA[aa] - ph))
    return charge


def compute_molecular_weight(sequence: str) -> float:
    """Compute molecular weight in Daltons."""
    water_mw = 18.02
    return sum(AA_MW.get(aa, 0) for aa in sequence) - (len(sequence) - 1) * water_mw


def compute_gravy(sequence: str) -> float:
    """Grand Average of Hydropathicity (Kyte-Doolittle)."""
    if not sequence:
        return 0.0
    return np.mean([KD_SCALE.get(aa, 0) for aa in sequence])


def compute_hydrophobic_moment(sequence: str, angle: float = 100.0) -> float:
    """Compute hydrophobic moment assuming alpha-helix (100 degree angle)."""
    if len(sequence) < 3:
        return 0.0
    angle_rad = np.radians(angle)
    sin_sum = 0.0
    cos_sum = 0.0
    for i, aa in enumerate(sequence):
        h = EISENBERG_SCALE.get(aa, 0)
        sin_sum += h * np.sin(i * angle_rad)
        cos_sum += h * np.cos(i * angle_rad)
    return np.sqrt(sin_sum**2 + cos_sum**2) / len(sequence)


def compute_isoelectric_point(sequence: str) -> float:
    """Binary search for isoelectric point."""
    low, high = 0.0, 14.0
    for _ in range(100):
        mid = (low + high) / 2
        charge = compute_net_charge(sequence, mid)
        if charge > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def compute_instability_index(sequence: str) -> float:
    """Guruprasad instability index (simplified)."""
    # Instability weight values for dipeptides (simplified)
    # Full table has 400 values; we use a heuristic approximation
    unstable_dipeptides = {
        "DG": 1, "DP": 1, "GD": 1, "GN": 1, "KD": 1,
        "NG": 1, "PD": 1, "QD": 1, "RD": 1, "SD": 1,
    }
    if len(sequence) < 2:
        return 0.0
    score = 0
    for i in range(len(sequence) - 1):
        dp = sequence[i : i + 2]
        if dp in unstable_dipeptides:
            score += 1
    # Normalize: real instability index is more complex, this is a proxy
    return (score / (len(sequence) - 1)) * 100


def compute_aromaticity(sequence: str) -> float:
    """Fraction of aromatic amino acids (F, W, Y)."""
    if not sequence:
        return 0.0
    aromatic = sum(1 for aa in sequence if aa in "FWY")
    return aromatic / len(sequence)


def compute_aliphatic_index(sequence: str) -> float:
    """Aliphatic index - thermostability indicator."""
    if not sequence:
        return 0.0
    n = len(sequence)
    ala = sequence.count("A") / n * 100
    val = sequence.count("V") / n * 100
    ile = sequence.count("I") / n * 100
    leu = sequence.count("L") / n * 100
    return ala + 2.9 * val + 3.9 * (ile + leu)


def compute_boman_index(sequence: str) -> float:
    """Boman index - protein binding potential (kcal/mol)."""
    boman_scale = {
        "L": -4.92, "I": -4.92, "V": -4.04, "F": -2.98, "M": -2.35,
        "W": -2.33, "A": -1.81, "C": -1.28, "G": -0.94, "Y": 0.14,
        "T": 0.26, "S": 0.84, "H": 2.06, "Q": 2.36, "K": 2.71,
        "N": 2.72, "E": 2.81, "D": 2.95, "R": 14.92, "P": 0.0,
    }
    if not sequence:
        return 0.0
    return sum(boman_scale.get(aa, 0) for aa in sequence) / len(sequence)


def compute_aa_composition(sequence: str) -> dict:
    """Compute amino acid composition as fractions."""
    n = len(sequence) if sequence else 1
    return {aa: sequence.count(aa) / n for aa in "ACDEFGHIKLMNPQRSTVWY"}


def compute_all_properties(sequence: str) -> dict:
    """Compute all biochemical properties for a peptide sequence."""
    # Clean sequence
    seq = "".join(c for c in sequence.upper() if c in STANDARD_AA)
    if len(seq) < 3:
        return None

    props = {
        "sequence": seq,
        "length": len(seq),
        "molecular_weight": round(compute_molecular_weight(seq), 1),
        "net_charge_ph7": round(compute_net_charge(seq, 7.0), 2),
        "isoelectric_point": round(compute_isoelectric_point(seq), 2),
        "gravy": round(compute_gravy(seq), 3),
        "hydrophobic_moment": round(compute_hydrophobic_moment(seq), 3),
        "instability_index": round(compute_instability_index(seq), 1),
        "aromaticity": round(compute_aromaticity(seq), 3),
        "aliphatic_index": round(compute_aliphatic_index(seq), 1),
        "boman_index": round(compute_boman_index(seq), 3),
    }

    # Add AA composition
    aa_comp = compute_aa_composition(seq)
    for aa, frac in aa_comp.items():
        props[f"aa_{aa}"] = round(frac, 4)

    return props


class PropertiesTool(BaseTool):
    name = "Biochemical Properties"
    description = "Computes biochemical properties of peptides"
    icon = "📊"

    def _execute(self, sequences: list[str]) -> ToolResult:
        if not sequences:
            return ToolResult(status="error", message="No sequences to analyze.")

        results = []
        for seq in sequences:
            props = compute_all_properties(seq)
            if props is not None:
                results.append(props)

        if not results:
            return ToolResult(status="error", message="No valid sequences for analysis.")

        df = pd.DataFrame(results)

        # Compute stats for message
        avg_charge = df["net_charge_ph7"].mean()
        avg_mw = df["molecular_weight"].mean()
        ideal_charge = ((df["net_charge_ph7"] >= 2) & (df["net_charge_ph7"] <= 9)).sum()
        ideal_hm = (df["hydrophobic_moment"] > 0.3).sum()

        msg = (
            f"Computed 12 biochemical properties for {len(results)} peptides. "
            f"Effective AMPs typically have charge +2 to +9 and high hydrophobic moment. "
            f"Of {len(results)} candidates: {ideal_charge} have charge in ideal range, "
            f"{ideal_hm} have high hydrophobic moment (amphipathicity). "
            f"Average molecular weight: {avg_mw:.0f} Da. Average charge: {avg_charge:.1f}."
        )

        return ToolResult(
            status="success",
            message=msg,
            data={
                "properties_df": df.to_dict("records"),
                "n_analyzed": len(results),
                "n_ideal_charge": int(ideal_charge),
                "n_ideal_hm": int(ideal_hm),
                "avg_charge": round(avg_charge, 2),
                "avg_mw": round(avg_mw, 0),
            },
        )
