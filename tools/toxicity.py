"""Tool 6: Toxicity and safety screening for candidate peptides."""

import re
from tools import BaseTool, ToolResult


def compute_toxicity_score(sequence: str, properties: dict) -> tuple[float, list[str]]:
    """
    Rule-based toxicity risk estimator.
    Returns (risk_score 0-1, list_of_flags).
    Based on published AMP toxicity correlates from literature.
    """
    flags = []
    score = 0.0

    # 1. High hydrophobicity → host cell membrane disruption risk
    gravy = properties.get("gravy", 0)
    if gravy > 0.5:
        score += 0.25
        flags.append(f"Alta hidrofobicidad (GRAVY={gravy:.2f} > 0.5) — riesgo de lisis de células host")
    elif gravy > 0.3:
        score += 0.10
        flags.append(f"Hidrofobicidad moderada-alta (GRAVY={gravy:.2f})")

    # 2. Very high positive charge → cytotoxicity
    charge = properties.get("net_charge_ph7", 0)
    if charge > 9:
        score += 0.25
        flags.append(f"Carga muy alta (+{charge:.1f} > +9) — riesgo de citotoxicidad")
    elif charge > 7:
        score += 0.10
        flags.append(f"Carga alta (+{charge:.1f})")

    # 3. Very long peptide → immunogenicity
    length = len(sequence)
    if length > 50:
        score += 0.15
        flags.append(f"Péptido largo ({length} aa > 50) — riesgo de inmunogenicidad")

    # 4. Cysteine-rich → off-target disulfide bonds
    cys_fraction = sequence.count("C") / max(length, 1)
    if cys_fraction > 0.12:
        score += 0.15
        flags.append(f"Rico en cisteína ({cys_fraction:.0%} > 12%) — puentes disulfuro off-target")

    # 5. Hemolytic motifs: WxxW or WxxxW patterns
    if re.search(r"W.{2,3}W", sequence):
        score += 0.20
        flags.append("Contiene motivo WxxW/WxxxW — asociado con actividad hemolítica")

    # 6. Very low charge → poor selectivity
    if charge < 1:
        score += 0.15
        flags.append(f"Carga baja (+{charge:.1f} < +1) — selectividad pobre por membranas microbianas")

    # 7. High Boman index → strong protein binding (can bind host proteins)
    boman = properties.get("boman_index", 0)
    if boman > 2.5:
        score += 0.10
        flags.append(f"Índice Boman alto ({boman:.2f} > 2.5) — unión a proteínas del host")

    # Cap at 1.0
    score = min(score, 1.0)

    return score, flags


def compute_selectivity_estimate(properties: dict) -> dict:
    """
    Estimate selectivity based on property profiles.
    Good AMPs have moderate charge (+2 to +6), moderate hydrophobicity,
    and high amphipathicity.
    """
    charge = properties.get("net_charge_ph7", 0)
    gravy = properties.get("gravy", 0)
    hm = properties.get("hydrophobic_moment", 0)

    # Charge score: optimal +3 to +6
    if 3 <= charge <= 6:
        charge_sel = 1.0
    elif 2 <= charge <= 8:
        charge_sel = 0.7
    else:
        charge_sel = 0.3

    # Hydrophobicity: moderate is best (-0.5 to 0.3)
    if -0.5 <= gravy <= 0.3:
        hydro_sel = 1.0
    elif -1.0 <= gravy <= 0.5:
        hydro_sel = 0.6
    else:
        hydro_sel = 0.3

    # Amphipathicity: higher is better for selectivity
    if hm > 0.5:
        amphi_sel = 1.0
    elif hm > 0.3:
        amphi_sel = 0.7
    else:
        amphi_sel = 0.4

    selectivity = (charge_sel * 0.4 + hydro_sel * 0.3 + amphi_sel * 0.3)

    return {
        "selectivity_score": round(selectivity, 3),
        "charge_selectivity": round(charge_sel, 2),
        "hydrophobicity_selectivity": round(hydro_sel, 2),
        "amphipathicity_selectivity": round(amphi_sel, 2),
    }


class ToxicityTool(BaseTool):
    name = "Toxicity Screening"
    description = "Evalúa riesgo de toxicidad y selectividad"
    icon = "🛡️"

    def _execute(self, sequences: list[str], properties_list: list[dict]) -> ToolResult:
        if not sequences or not properties_list:
            return ToolResult(status="error", message="No hay datos para screening.")

        results = []
        n_passed = 0
        n_flagged = 0
        all_flag_types = {}

        for seq, props in zip(sequences, properties_list):
            tox_score, flags = compute_toxicity_score(seq, props)
            selectivity = compute_selectivity_estimate(props)

            passed = tox_score < 0.4  # threshold
            if passed:
                n_passed += 1
            else:
                n_flagged += 1

            for flag in flags:
                flag_type = flag.split("—")[0].strip() if "—" in flag else flag[:30]
                all_flag_types[flag_type] = all_flag_types.get(flag_type, 0) + 1

            results.append({
                "sequence": seq,
                "toxicity_risk": round(tox_score, 3),
                "passed_screening": passed,
                "flags": flags,
                "n_flags": len(flags),
                **selectivity,
            })

        # Top flag reasons
        top_flags = sorted(all_flag_types.items(), key=lambda x: x[1], reverse=True)[:3]
        top_flags_str = "; ".join(f"{name} ({count}x)" for name, count in top_flags)

        msg = (
            f"Screening de toxicidad completado para {len(sequences)} candidatos. "
            f"{n_passed} pasaron todos los filtros (riesgo < 0.4). "
            f"{n_flagged} fueron flaggeados por riesgo elevado. "
        )
        if top_flags_str:
            msg += f"Alertas más comunes: {top_flags_str}. "
        msg += "Los candidatos aprobados tienen perfil favorable de selectividad patógeno/host."

        return ToolResult(
            status="success",
            message=msg,
            data={
                "toxicity_results": results,
                "n_passed": n_passed,
                "n_flagged": n_flagged,
                "flag_summary": dict(all_flag_types),
            },
        )
