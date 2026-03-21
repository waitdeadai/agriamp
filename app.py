"""AgriAMP — AI-Powered Antimicrobial Peptide Discovery for Crop Protection"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
import os
import sys


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lang import STRINGS, PATHOGEN_INFO as PATHOGEN_INFO_I18N

# Detect cloud mode (no GPU / no torch available)
def _check_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

IS_CLOUD = not _check_gpu()
PRECOMPUTED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "precomputed")

# ── Page Config ──
st.set_page_config(
    page_title="AgriAMP — Biopesticide Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a5e1a, #2d8f2d, #4caf50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #888;
        margin-top: -8px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f0f7f0;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #2d8f2d;
    }
    .step-success { border-left: 3px solid #28a745; padding-left: 10px; }
    .step-running { border-left: 3px solid #ffc107; padding-left: 10px; }
    .step-error { border-left: 3px solid #dc3545; padding-left: 10px; }
    div[data-testid="stStatusWidget"] { display: none; }
    .hero-box {
        background: linear-gradient(135deg, #0d2818 0%, #1a3a2a 100%);
        border: 1px solid #2d8f2d44;
        border-radius: 12px;
        padding: 24px 28px;
        margin: 8px 0 20px 0;
        color: #e0e0e0;
    }
    .hero-box h3 { color: #4caf50; margin-bottom: 8px; }
    .hero-stat {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4caf50;
        line-height: 1.2;
    }
    .hero-label {
        font-size: 0.78rem;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ── Language + Pathogen info ──
def _get_lang():
    """Get current language from session state."""
    return st.session_state.get("lang", "en")

def _t(key):
    """Get translated string."""
    return STRINGS[_get_lang()].get(key, key)

def _pinfo():
    """Get pathogen info for current language."""
    return PATHOGEN_INFO_I18N[_get_lang()]


def render_sidebar():
    """Render sidebar with controls."""
    with st.sidebar:
        # Language toggle
        if "lang" not in st.session_state:
            st.session_state.lang = "en"
        lang_choice = st.radio("🌐", ["EN", "ES"], horizontal=True, label_visibility="collapsed",
                               index=0 if st.session_state.lang == "en" else 1)
        st.session_state.lang = lang_choice.lower()
        t = STRINGS[st.session_state.lang]
        pinfo = _pinfo()

        st.markdown(f"### {t['analysis_config']}")

        pathogen = st.selectbox(
            t["target_pathogen"],
            list(pinfo.keys()),
            format_func=lambda x: f"{x} ({pinfo[x]['common']})",
        )

        info = pinfo[pathogen]
        st.markdown(f"""
        **{t['type']}:** {info['type']}
        **{t['crops']}:** {info['crops']}
        **{t['region']}:** {info['region']}
        **{t['impact']}:** {info['impact']}
        """)

        st.divider()

        if not IS_CLOUD:
            with st.expander(t["advanced_params"], expanded=False):
                max_variants = st.slider(t["max_variants"], 50, 300, 150, 25)
                tox_threshold = st.slider(t["tox_threshold"], 0.1, 0.8, 0.4, 0.05)
        else:
            max_variants = 150
            tox_threshold = 0.4

        st.divider()

        precomputed_path = os.path.join(
            PRECOMPUTED_DIR, f"{pathogen.replace(' ', '_').lower()}.json"
        )

        if IS_CLOUD:
            run_button = False
            load_precomputed = os.path.exists(precomputed_path)
            st.info(t["cloud_mode"])
        else:
            run_button = st.button(
                t["run_pipeline"],
                type="primary",
                use_container_width=True,
            )
            load_precomputed = False
            if os.path.exists(precomputed_path):
                load_precomputed = st.checkbox(t["load_precomputed"], value=False)

        st.divider()
        with st.expander(t["what_are_amps"], expanded=False):
            st.markdown(t["amps_explanation"])
        with st.expander(t["how_agriamp_works"], expanded=False):
            st.markdown(t["agriamp_explanation"])
        st.divider()
        st.markdown("""
        **AgriAMP** v1.0
        Aleph Hackathon M26 — Track Biotech
        [GitHub](https://github.com/waitdeadai/agriamp)
        """)

        return pathogen, run_button, load_precomputed, precomputed_path, max_variants, tox_threshold



def render_top_candidates(df: pd.DataFrame):
    """Tab 1: Top candidates dashboard."""
    t = STRINGS[_get_lang()]
    if df.empty:
        st.warning(t["no_candidates"])
        return

    # Filter to passed toxicity
    passed = df[df["passed_toxicity"]].head(20)
    all_ranked = df.head(30)
    display_df = passed if len(passed) >= 5 else all_ranked

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(t["candidates_analyzed"], len(df))
    with col2:
        st.metric(t["passed_toxicity"], len(df[df["passed_toxicity"]]))
    with col3:
        st.metric("Top AgriAMP Score", f"{df['agriamp_score'].max():.3f}")
    with col4:
        avg_charge = display_df["net_charge"].mean()
        st.metric(t["avg_charge"], f"+{avg_charge:.1f}")

    st.divider()

    # Top 10 bar chart
    top10 = display_df.head(10).copy()
    top10["label"] = top10.apply(
        lambda r: r["name"] if r["name"] else r["sequence"][:12] + "...", axis=1
    )

    fig = px.bar(
        top10,
        x="label",
        y="agriamp_score",
        color="toxicity_risk",
        color_continuous_scale="RdYlGn_r",
        labels={"agriamp_score": "AgriAMP Score", "label": t["peptide"], "toxicity_risk": t["tox_risk"]},
        title=t["top10_title"],
    )
    fig.add_hline(y=0.7, line_dash="dash", line_color="#4caf50", opacity=0.5,
                  annotation_text=t["threshold_line"])
    fig.update_layout(height=400, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader(t["candidate_table"])
    table_cols = [
        "sequence", "agriamp_score", "amp_probability", "net_charge",
        "molecular_weight", "hydrophobic_moment", "toxicity_risk",
        "passed_toxicity", "origin", "name",
    ]
    available_cols = [c for c in table_cols if c in display_df.columns]
    st.dataframe(
        display_df[available_cols].style.format({
            "agriamp_score": "{:.4f}",
            "amp_probability": "{:.4f}",
            "net_charge": "{:+.2f}",
            "molecular_weight": "{:.0f}",
            "hydrophobic_moment": "{:.4f}",
            "toxicity_risk": "{:.3f}",
        }).background_gradient(subset=["agriamp_score"], cmap="Greens"),
        use_container_width=True,
        height=400,
    )


def render_property_analysis(df: pd.DataFrame, metrics: dict):
    """Tab 2: Property analysis with radar charts and distributions."""
    t = STRINGS[_get_lang()]
    if df.empty:
        st.warning(t["no_data"])
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(t["radar_title"])

        # Ideal AMP profile (normalized 0-1)
        ideal = {
            t["charge"]: 0.7,
            t["amphipathicity"]: 0.8,
            t["stability"]: 0.7,
            t["selectivity"]: 0.8,
            t["low_toxicity"]: 0.9,
        }

        top = df.iloc[0]
        candidate = {
            t["charge"]: min(max(top["net_charge"] / 6, 0), 1),
            t["amphipathicity"]: min(max(top["hydrophobic_moment"] / 0.8, 0), 1),
            t["stability"]: max(1 - top["instability_index"] / 100, 0),
            t["selectivity"]: top.get("selectivity_score", 0.5),
            t["low_toxicity"]: 1 - top["toxicity_risk"],
        }

        categories = list(ideal.keys())

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(ideal.values()) + [list(ideal.values())[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=t["ideal_amp_profile"],
            opacity=0.3,
            line=dict(color="#2d8f2d"),
        ))
        fig.add_trace(go.Scatterpolar(
            r=list(candidate.values()) + [list(candidate.values())[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=f"Top: {top.get('name', top['sequence'][:10])}",
            opacity=0.5,
            line=dict(color="#1a5e1a"),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(t["feature_importance"])
        top_features = metrics.get("top_features", [])
        if top_features:
            feat_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
            fig2 = px.bar(
                feat_df.head(10),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Greens",
                title=t["top10_features"],
            )
            fig2.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(t["no_feature_importance"])

    # Distribution plots
    st.subheader(t["property_distributions"])
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.histogram(df, x="net_charge", nbins=30, title=t["net_charge_title"],
                           color_discrete_sequence=["#2d8f2d"])
        fig.add_vline(x=2, line_dash="dash", line_color="red", annotation_text=t["min_ideal"])
        fig.add_vline(x=9, line_dash="dash", line_color="red", annotation_text=t["max_ideal"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="hydrophobic_moment", nbins=30, title=t["hydrophobic_moment_title"],
                           color_discrete_sequence=["#1a5e1a"])
        fig.add_vline(x=0.3, line_dash="dash", line_color="red", annotation_text=t["amphipathicity_threshold"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = px.histogram(df, x="molecular_weight", nbins=30, title=t["molecular_weight_title"],
                           color_discrete_sequence=["#4caf50"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def render_sequence_viewer(df: pd.DataFrame):
    """Tab 3: Interactive sequence viewer with colored amino acids."""
    t = STRINGS[_get_lang()]
    if df.empty:
        st.warning(t["no_sequences"])
        return

    top_candidates = df.head(10)
    selected_idx = st.selectbox(
        t["select_peptide"],
        range(len(top_candidates)),
        format_func=lambda i: f"#{i+1} — {top_candidates.iloc[i].get('name', '')} "
                              f"({top_candidates.iloc[i]['sequence'][:20]}...) "
                              f"Score: {top_candidates.iloc[i]['agriamp_score']:.3f}",
    )

    pep = top_candidates.iloc[selected_idx]
    seq = pep["sequence"]

    # Color-coded sequence
    aa_colors = {
        # Hydrophobic (yellow)
        "A": "#FFE082", "V": "#FFD54F", "L": "#FFCA28", "I": "#FFC107",
        "M": "#FFB300", "F": "#FFA000", "W": "#FF8F00", "P": "#FFE082",
        # Positively charged (blue)
        "K": "#64B5F6", "R": "#42A5F5", "H": "#90CAF9",
        # Negatively charged (red)
        "D": "#EF9A9A", "E": "#E57373",
        # Polar (green)
        "S": "#A5D6A7", "T": "#81C784", "N": "#66BB6A", "Q": "#4CAF50",
        "C": "#AED581", "G": "#C8E6C9", "Y": "#B2DFDB",
    }

    # Render colored sequence
    html_seq = ""
    for i, aa in enumerate(seq):
        color = aa_colors.get(aa, "#E0E0E0")
        html_seq += f'<span style="background-color:{color};padding:2px 4px;margin:1px;font-family:monospace;font-size:16px;border-radius:3px;font-weight:bold;" title="{aa} (pos {i+1})">{aa}</span>'

    st.markdown(f"### {t['colored_sequence']}")
    st.markdown(f"<div style='line-height:2.5;'>{html_seq}</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:10px;font-size:12px;">
    <span style="background-color:#FFCA28;padding:2px 6px;border-radius:3px;">{t['hydrophobic']}</span>
    <span style="background-color:#42A5F5;padding:2px 6px;border-radius:3px;color:white;">{t['cationic']}</span>
    <span style="background-color:#E57373;padding:2px 6px;border-radius:3px;">{t['anionic']}</span>
    <span style="background-color:#66BB6A;padding:2px 6px;border-radius:3px;">{t['polar']}</span>
    </div>
    """, unsafe_allow_html=True)

    # Property cards
    st.markdown(f"### {t['properties']}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(t["net_charge"], f"+{pep['net_charge']:.1f}")
        st.metric(t["molecular_weight"], f"{pep['molecular_weight']:.0f} Da")
    with col2:
        st.metric(t["hydrophobic_moment"], f"{pep['hydrophobic_moment']:.3f}")
        st.metric("GRAVY", f"{pep['gravy']:.3f}")
    with col3:
        st.metric(t["isoelectric_point"], f"{pep['isoelectric_point']:.1f}")
        st.metric(t["boman_index"], f"{pep['boman_index']:.2f}")
    with col4:
        st.metric(t["toxicity"], f"{pep['toxicity_risk']:.2f}")
        st.metric("AgriAMP Score", f"{pep['agriamp_score']:.4f}")

    # Score breakdown
    st.markdown(f"### {t['score_composition']}")
    _charge_norm = min(max(pep["net_charge"] / 6, 0), 1)
    _hm_norm = min(max(pep["hydrophobic_moment"] / 0.8, 0), 1)
    _stab_norm = max(1 - pep["instability_index"] / 100, 0)
    _tox_comp = 1 - pep["toxicity_risk"]
    _components = {
        t["amp_prob_component"]: pep["amp_probability"] * 0.35,
        t["charge_component"]: _charge_norm * 0.25,
        t["amphipathicity_component"]: _hm_norm * 0.20,
        t["stability_component"]: _stab_norm * 0.10,
        t["low_tox_component"]: _tox_comp * 0.10,
    }
    _comp_df = pd.DataFrame({t["component"]: list(_components.keys()), t["contribution"]: list(_components.values())})
    _fig_comp = px.bar(_comp_df, x=t["contribution"], y=t["component"], orientation="h",
                       color=t["contribution"], color_continuous_scale="Greens",
                       title=f"Score total: {pep['agriamp_score']:.4f}")
    _fig_comp.update_layout(height=250, showlegend=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(_fig_comp, use_container_width=True)

    # Helical wheel projection
    st.markdown(f"### {t['helical_wheel']}")
    st.caption(t["helical_caption"])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    angle_step = 100  # degrees for alpha helix

    max_residues = min(len(seq), 18)
    for i in range(max_residues):
        angle_rad = np.radians(i * angle_step)
        r = 1.0
        x = r * np.cos(angle_rad)
        y = r * np.sin(angle_rad)

        aa = seq[i]
        if aa in "KRH":
            color = "#42A5F5"
        elif aa in "AVLIFMWP":
            color = "#FFCA28"
        elif aa in "DE":
            color = "#E57373"
        else:
            color = "#66BB6A"

        circle = plt.Circle((x, y), 0.15, color=color, ec="black", lw=1.5, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, aa, ha="center", va="center", fontsize=10, fontweight="bold", zorder=3)

        # Draw line from center
        ax.plot([0, x * 0.85], [0, y * 0.85], "k-", alpha=0.2, lw=0.5, zorder=1)

        # Number label
        ax.text(x * 1.25, y * 1.25, str(i + 1), ha="center", va="center", fontsize=7, color="gray")

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Helical Wheel — {pep.get('name', seq[:10])}", fontsize=12)
    st.pyplot(fig)
    plt.close()


def render_validation(metrics: dict):
    """Tab 4: ML validation metrics."""
    t = STRINGS[_get_lang()]
    if not metrics:
        st.warning(t["no_validation"])
        return

    cv_auc = metrics.get("cv_auc_mean")

    if cv_auc is None:
        st.info(t["property_scoring_info"])
        return

    # Comprehensive metrics row
    st.subheader(t["classifier_metrics"])
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    with col_m1:
        st.metric("AUC-ROC", f"{cv_auc:.3f}")
    with col_m2:
        st.metric("MCC", f"{metrics.get('oof_mcc', 0):.3f}")
    with col_m3:
        st.metric("Accuracy", f"{metrics.get('oof_accuracy', 0):.3f}")
    with col_m4:
        st.metric("F1-Score", f"{metrics.get('oof_f1', 0):.3f}")
    with col_m5:
        st.metric(t["sensitivity"], f"{metrics.get('oof_sensitivity', 0):.3f}")
    with col_m6:
        st.metric(t["specificity"], f"{metrics.get('oof_specificity', 0):.3f}")

    st.caption(
        t["data_caption"].format(
            pos=metrics.get('n_train_positive', 0),
            neg=metrics.get('n_train_negative', 0),
            std=metrics.get('cv_auc_std', 0),
        )
    )

    col1, col2 = st.columns(2)

    with col1:
        # Confusion matrix heatmap
        cm = metrics.get("confusion_matrix")
        if cm and len(cm) == 2:
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Pred: non-AMP", "Pred: AMP"],
                y=["Real: non-AMP", "Real: AMP"],
                text=[[str(cm[0][0]), str(cm[0][1])],
                      [str(cm[1][0]), str(cm[1][1])]],
                texttemplate="%{text}",
                colorscale="Greens",
                showscale=False,
            ))
            fig_cm.update_layout(
                title=t["confusion_matrix"],
                height=350,
                xaxis_title=t["prediction"],
                yaxis_title=t["actual"],
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # CV scores bar
        cv_scores = metrics.get("cv_scores", [])
        if cv_scores:
            fig = px.bar(
                x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                y=cv_scores,
                labels={"x": "Fold", "y": "AUC-ROC"},
                title=t["auc_per_fold"],
                color=cv_scores,
                color_continuous_scale="Greens",
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROC curve (use OOF predictions if available, else training)
        y_true = metrics.get("y_train_true", [])
        y_pred_oof = metrics.get("y_oof_proba", [])
        y_pred_train = metrics.get("y_train_pred", [])

        if y_true and (y_pred_oof or y_pred_train):
            from sklearn.metrics import roc_curve, auc

            fig = go.Figure()

            if y_pred_oof:
                fpr_oof, tpr_oof, _ = roc_curve(y_true, y_pred_oof)
                auc_oof = auc(fpr_oof, tpr_oof)
                fig.add_trace(go.Scatter(
                    x=fpr_oof, y=tpr_oof, mode="lines",
                    name=f"OOF (AUC={auc_oof:.3f})",
                    line=dict(color="#2d8f2d", width=2),
                ))

            if y_pred_train:
                fpr_tr, tpr_tr, _ = roc_curve(y_true, y_pred_train)
                auc_tr = auc(fpr_tr, tpr_tr)
                fig.add_trace(go.Scatter(
                    x=fpr_tr, y=tpr_tr, mode="lines",
                    name=f"Training (AUC={auc_tr:.3f})",
                    line=dict(color="#81C784", width=1.5, dash="dot"),
                ))

            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random",
                line=dict(color="gray", dash="dash"),
            ))
            fig.update_layout(
                title=t["roc_curve"],
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_caso_real(df: pd.DataFrame, pathogen: str):
    """Tab: Real Case — real case study with verified data."""
    t = STRINGS[_get_lang()]
    st.markdown(t["caso_real_hero"], unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t["loss_botrytis"], "50-80%", delta=None, help=t["loss_botrytis_help"])
        st.metric(t["wine_exports"], "~$800M/yr", help=t["wine_exports_help"])
    with col2:
        st.metric(t["global_losses"], "~$2B USD", delta=None, help=t["global_losses_help"])
        st.metric(t["agrochemicals"], "500M+ L/kg", help=t["agrochemicals_help"])
    with col3:
        st.metric(t["weight_loss"], "53%", delta=None, help=t["weight_loss_help"])
        st.metric(t["banned_pesticides"], "106", help=t["banned_pesticides_help"])

    st.divider()

    st.markdown(f"#### {t['regulatory_crisis']}")
    st.markdown(t["regulatory_content"])

    st.divider()

    st.markdown(f"#### {t['validated_amps']}")
    st.markdown(t["validated_amps_table"])

    # Show Epinecidin in results if present
    if not df.empty:
        epi_matches = df[df["name"].str.contains("Epinecidin|EPI-4", case=False, na=False)]
        if not epi_matches.empty:
            st.divider()
            st.markdown(f"#### {t['epinecidin_in_results']}")
            for _, row in epi_matches.iterrows():
                status = t["approved"] if row["passed_toxicity"] else t["flagged"]
                st.markdown(
                    t["epinecidin_row"].format(
                        name=row['name'], score=row['agriamp_score'],
                        charge=row['net_charge'], prob=row['amp_probability'],
                        tox=row['toxicity_risk'], status=status,
                    )
                )

    st.divider()
    st.markdown(f"#### {t['next_step']}")
    st.markdown(t["next_step_content"])


def render_benchmark(metrics: dict):
    """Tab: Benchmark comparison vs published SOTA tools."""
    t = STRINGS[_get_lang()]
    st.markdown(f"### {t['benchmark_title']}")
    st.markdown(t["benchmark_description"])

    # Our metrics
    our_auc = metrics.get("cv_auc_mean", 0)
    our_mcc = metrics.get("oof_mcc", 0)
    our_sens = metrics.get("oof_sensitivity", 0)
    our_spec = metrics.get("oof_specificity", 0)
    our_acc = metrics.get("oof_accuracy", 0)
    our_f1 = metrics.get("oof_f1", 0)
    n_train = metrics.get("n_train_positive", 0) + metrics.get("n_train_negative", 0)

    # Comparison data
    benchmark_data = [
        {
            t["tool"]: t["agriamp_ours"],
            t["year"]: "2026",
            t["method"]: "ESM-2 + RF",
            t["dataset"]: f"{n_train} seqs",
            "AUC": our_auc,
            "MCC": our_mcc,
            t["sensitivity"]: our_sens,
            t["specificity"]: our_spec,
            t["ref"]: t["this_work"],
        },
        {
            t["tool"]: "amPEPpy",
            t["year"]: "2020",
            t["method"]: "RF + features",
            t["dataset"]: "Custom",
            "AUC": 0.99,
            "MCC": 0.90,
            t["sensitivity"]: None,
            t["specificity"]: None,
            t["ref"]: "Bioinformatics, 2021",
        },
        {
            t["tool"]: "PLAPD",
            t["year"]: "2025",
            t["method"]: "ESM-2 + CNN + Transformer",
            t["dataset"]: "8,268 seqs",
            "AUC": 0.922,
            "MCC": 0.749,
            t["sensitivity"]: None,
            t["specificity"]: 0.946,
            t["ref"]: "Methods, 2025",
        },
        {
            t["tool"]: "AMP-RNNpro",
            t["year"]: "2024",
            t["method"]: "RNN + prob. features",
            t["dataset"]: "Custom",
            "AUC": None,
            "MCC": None,
            t["sensitivity"]: 0.965,
            t["specificity"]: 0.979,
            t["ref"]: "Sci. Reports, 2024",
        },
        {
            t["tool"]: "sAMPpred-GAT",
            t["year"]: "2023",
            t["method"]: "GAT + structure",
            t["dataset"]: "Multiple",
            "AUC": None,
            "MCC": None,
            t["sensitivity"]: None,
            t["specificity"]: None,
            t["ref"]: "8 test sets, 2023",
        },
    ]

    bench_df = pd.DataFrame(benchmark_data)

    def fmt_metric(v):
        if v is None:
            return "—"
        return f"{v:.3f}"

    display_df = bench_df.copy()
    for col in ["AUC", "MCC", t["sensitivity"], t["specificity"]]:
        display_df[col] = display_df[col].apply(fmt_metric)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption(t["benchmark_note"])

    # AUC comparison bar chart
    auc_tools = [d for d in benchmark_data if d["AUC"] is not None]
    if auc_tools:
        fig = go.Figure()
        colors = ["#2d8f2d" if d[t["tool"]].startswith("AgriAMP") else "#666"
                  for d in auc_tools]
        fig.add_trace(go.Bar(
            x=[d[t["tool"]] for d in auc_tools],
            y=[d["AUC"] for d in auc_tools],
            marker_color=colors,
            text=[f"{d['AUC']:.3f}" for d in auc_tools],
            textposition="outside",
        ))
        fig.update_layout(
            title=t["auc_chart_title"],
            yaxis_title="AUC-ROC",
            yaxis=dict(range=[0.85, 1.02]),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown(f"#### {t['differentiator_title']}")
    st.markdown(t["differentiator_content"])

    # Our full metrics summary
    st.divider()
    st.markdown(f"#### {t['full_metrics']}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AUC-ROC", f"{our_auc:.3f}")
        st.metric("MCC", f"{our_mcc:.3f}")
    with col2:
        st.metric("Accuracy", f"{our_acc:.3f}")
        st.metric("F1-Score", f"{our_f1:.3f}")
    with col3:
        st.metric(t["sensitivity"], f"{our_sens:.3f}")
        st.metric(t["specificity"], f"{our_spec:.3f}")
    with col4:
        st.metric(t["training_data"], f"{n_train}")
        st.metric(t["cv_folds"], t["stratified"])


def render_export(df: pd.DataFrame, pathogen: str):
    """Tab 5: Export results."""
    t = STRINGS[_get_lang()]
    if df.empty:
        st.warning(t["no_export"])
        return

    st.subheader(t["export_results"])

    col1, col2 = st.columns(2)

    with col1:
        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            t["download_csv"],
            data=csv,
            file_name=f"agriamp_{pathogen.replace(' ', '_').lower()}_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        # FASTA download
        fasta_lines = []
        passed = df[df["passed_toxicity"]].head(20)
        for _, row in passed.iterrows():
            name = row.get("name", "candidate")
            header = f">{name}|score={row['agriamp_score']:.4f}|charge={row['net_charge']:+.1f}|MW={row['molecular_weight']:.0f}"
            fasta_lines.append(header)
            fasta_lines.append(row["sequence"])

        fasta_text = "\n".join(fasta_lines)
        st.download_button(
            t["download_fasta"],
            data=fasta_text,
            file_name=f"agriamp_{pathogen.replace(' ', '_').lower()}_top.fasta",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()
    st.subheader(t["analysis_summary"])
    st.json({
        "pathogen": pathogen,
        "total_candidates": len(df),
        "passed_toxicity": int(df["passed_toxicity"].sum()),
        "top_score": float(df["agriamp_score"].max()),
        "avg_score": float(df["agriamp_score"].mean()),
        "avg_charge": float(df["net_charge"].mean()),
        "avg_mw": float(df["molecular_weight"].mean()),
    })


# ── MAIN APP ──
def main():
    # Header with logo
    _logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo_agriamp.png")
    _hcol1, _hcol2 = st.columns([0.08, 0.92])
    with _hcol1:
        if os.path.exists(_logo_path):
            st.image(_logo_path, width=64)
    with _hcol2:
        st.markdown('<p class="main-header">AgriAMP</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="subtitle">{_t("subtitle")}</p>', unsafe_allow_html=True)

    # Sidebar
    pathogen, run_button, load_precomputed, precomputed_path, max_variants, tox_threshold = render_sidebar()

    # Initialize session state
    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None
    if "agent_steps" not in st.session_state:
        st.session_state.agent_steps = []
    if "loaded_pathogen" not in st.session_state:
        st.session_state.loaded_pathogen = None

    # Load precomputed if requested (or auto-load on cloud)
    _should_load = load_precomputed and os.path.exists(precomputed_path)
    if not _should_load and IS_CLOUD and os.path.exists(precomputed_path):
        # Auto-load on cloud: first visit or pathogen changed
        if st.session_state.agent_result is None or st.session_state.loaded_pathogen != pathogen:
            _should_load = True

    if _should_load and os.path.exists(precomputed_path):
        with open(precomputed_path, "r") as f:
            cached = json.load(f)
        st.session_state.agent_result = {
            "candidates": pd.DataFrame(cached["candidates"]),
            "metrics": cached.get("metrics", {}),
            "steps": cached.get("steps", []),
            "total_duration": cached.get("total_duration", 0),
        }
        st.session_state.loaded_pathogen = pathogen
        if not IS_CLOUD:
            st.success(_t("precomputed_loaded"))

    # Run pipeline
    if run_button:
        from agent import AgriAMPAgent, AgentStep

        @st.cache_resource
        def get_agent():
            return AgriAMPAgent()

        agent = get_agent()

        # Agent workflow display
        status_container = st.container()
        with status_container:
            st.subheader(_t("agentic_workflow"))
            step_placeholder = st.empty()
            steps_display = []

            def on_step(step: AgentStep):
                steps_display.append(step)
                with step_placeholder.container():
                    for s in steps_display:
                        icon = {"success": "✅", "warning": "⚠️", "error": "❌", "running": "⏳"}.get(s.status, "⏳")
                        with st.expander(f"{s.icon} {s.tool_name} {icon}", expanded=(s.status == "running")):
                            st.markdown(s.message)
                            if s.duration > 0:
                                st.caption(f"{_t('duration')}: {s.duration:.1f}s")

            with st.spinner(_t("running_pipeline")):
                result = agent.run(pathogen=pathogen, callback=on_step)

            if result.success:
                st.session_state.agent_result = {
                    "candidates": result.candidates,
                    "metrics": result.metrics,
                    "steps": [(s.tool_name, s.icon, s.status, s.message, s.duration) for s in result.steps],
                    "total_duration": result.total_duration,
                }
                st.success(_t("pipeline_completed").format(dur=result.total_duration, n=len(result.candidates)))

                # Save precomputed results
                try:
                    cache_data = {
                        "candidates": result.candidates.to_dict("records"),
                        "metrics": {k: v for k, v in result.metrics.items()
                                    if isinstance(v, (int, float, str, list, dict, bool, type(None)))},
                        "steps": [(s.tool_name, s.icon, s.status, s.message, s.duration) for s in result.steps],
                        "total_duration": result.total_duration,
                    }
                    cache_path = os.path.join(
                        os.path.dirname(__file__), "data", "precomputed",
                        f"{pathogen.replace(' ', '_').lower()}.json"
                    )
                    with open(cache_path, "w") as f:
                        json.dump(cache_data, f, default=str)
                except Exception:
                    pass  # Non-critical
            else:
                st.error(_t("pipeline_error"))

    # Display results
    if st.session_state.agent_result is not None:
        data = st.session_state.agent_result
        df = data["candidates"] if isinstance(data["candidates"], pd.DataFrame) else pd.DataFrame(data["candidates"])
        metrics = data.get("metrics", {})

        # Show agent workflow log
        steps = data.get("steps", [])
        if steps:
            with st.expander(_t("workflow_steps_completed"), expanded=False):
                for step in steps:
                    if isinstance(step, (list, tuple)) and len(step) >= 4:
                        name, icon, status, msg = step[0], step[1], step[2], step[3]
                        dur = step[4] if len(step) > 4 else 0
                    else:
                        continue
                    status_icon = {"success": ":green[OK]", "warning": ":orange[WARN]", "error": ":red[ERR]"}.get(status, "?")
                    dur_str = f" *({dur:.1f}s)*" if dur and dur > 0 else ""
                    st.markdown(f"{icon} **{name}** {status_icon}{dur_str}  \n{msg}")

            _total_dur = data.get('total_duration', 0)
            _n_cands = len(df)
            _n_passed = int(df['passed_toxicity'].sum()) if 'passed_toxicity' in df.columns else 0
            st.caption(_t("pipeline_summary").format(dur=_total_dur, n_cands=_n_cands, n_passed=_n_passed))

        _pi = _pinfo()[pathogen]
        _t_str = STRINGS[_get_lang()]
        st.markdown(_t_str["results_for"].format(pathogen=pathogen, common=_pi['common']))
        st.caption(f"{_pi['type']} | {_t_str['crops']}: {_pi['crops']} | {_pi['impact']}")

        st.divider()

        # Results tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            f"🌍 {_t('tab_real_case')}",
            f"🏆 {_t('tab_top_candidates')}",
            f"📊 {_t('tab_property_analysis')}",
            f"🧬 {_t('tab_sequence_viewer')}",
            f"📈 {_t('tab_benchmark')}",
            f"✅ {_t('tab_ml_validation')}",
            f"📥 {_t('tab_export')}",
        ])

        with tab1:
            render_caso_real(df, pathogen)
        with tab2:
            render_top_candidates(df)
        with tab3:
            render_property_analysis(df, metrics)
        with tab4:
            render_sequence_viewer(df)
        with tab5:
            render_benchmark(metrics)
        with tab6:
            render_validation(metrics)
        with tab7:
            render_export(df, pathogen)

    else:
        # Welcome screen — hero
        _ts = STRINGS[_get_lang()]
        st.markdown(f"""
        <div class="hero-box">
            <h3>{_ts['hero_title']}</h3>
            <p style="font-size:1.05rem;margin-bottom:16px;">
                {_ts['hero_description']}
            </p>
            <div style="display:flex;gap:40px;flex-wrap:wrap;">
                <div><div class="hero-stat">650M</div><div class="hero-label">{_ts['hero_esm2']}</div></div>
                <div><div class="hero-stat">0.977</div><div class="hero-label">{_ts['hero_auc']}</div></div>
                <div><div class="hero-stat">4,600+</div><div class="hero-label">{_ts['hero_seqs']}</div></div>
                <div><div class="hero-stat">6</div><div class="hero-label">{_ts['hero_tools']}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"#### {_ts['problem']}")
            st.markdown(_ts["problem_desc"])
        with col2:
            st.markdown(f"#### {_ts['solution']}")
            st.markdown(_ts["solution_desc"])
        with col3:
            st.markdown(f"#### {_ts['agentic_pipeline']}")
            st.markdown(_ts["pipeline_desc"])

        st.info(_ts["select_pathogen_info"])


if __name__ == "__main__":
    main()
