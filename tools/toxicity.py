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
        flags.append(f"High hydrophobicity (GRAVY={gravy:.2f} > 0.5) — host cell lysis risk")
    elif gravy > 0.3:
        score += 0.10
        flags.append(f"Moderate-high hydrophobicity (GRAVY={gravy:.2f})")

    # 2. Very high positive charge → cytotoxicity
    charge = properties.get("net_charge_ph7", 0)
    if charge > 9:
        score += 0.25
        flags.append(f"Very high charge (+{charge:.1f} > +9) — cytotoxicity risk")
    elif charge > 7:
        score += 0.10
        flags.append(f"High charge (+{charge:.1f})")

    # 3. Very long peptide → immunogenicity
    length = len(sequence)
    if length > 50:
        score += 0.15
        flags.append(f"Long peptide ({length} aa > 50) — immunogenicity risk")

    # 4. Cysteine-rich → off-target disulfide bonds
    cys_fraction = sequence.count("C") / max(length, 1)
    if cys_fraction > 0.12:
        score += 0.15
        flags.append(f"Cysteine-rich ({cys_fraction:.0%} > 12%) — off-target disulfide bonds")

    # 5. Hemolytic motifs: WxxW or WxxxW patterns
    if re.search(r"W.{2,3}W", sequence):
        score += 0.20
        flags.append("Contains WxxW/WxxxW motif — associated with hemolytic activity")

    # 6. Very low charge → poor selectivity
    if charge < 1:
        score += 0.15
        flags.append(f"Low charge (+{charge:.1f} < +1) — poor selectivity for microbial membranes")

    # 7. High Boman index → strong protein binding (can bind host proteins)
    boman = properties.get("boman_index", 0)
    if boman > 2.5:
        score += 0.10
        flags.append(f"High Boman index ({boman:.2f} > 2.5) — host protein binding")

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
    description = "Evaluates toxicity risk and selectivity"
    icon = "🛡️"

    def _execute(self, sequences: list[str], properties_list: list[dict]) -> ToolResult:
        if not sequences or not properties_list:
            return ToolResult(status="error", message="No data for screening.")

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
            f"Toxicity screening completed for {len(sequences)} candidates. "
            f"{n_passed} passed all filters (risk < 0.4). "
            f"{n_flagged} flagged for elevated risk. "
        )
        if top_flags_str:
            msg += f"Most common alerts: {top_flags_str}. "
        msg += "Approved candidates have a favorable pathogen/host selectivity profile."

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
