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

# ── Pathogen info ──
PATHOGEN_INFO = {
    "Botrytis cinerea": {
        "common": "Podredumbre gris",
        "crops": "Uvas, frutillas, tomates",
        "region": "Mendoza, Patagonia",
        "impact": "Hasta 30% de perdida en temporadas humedas",
        "type": "Hongo",
    },
    "Fusarium graminearum": {
        "common": "Fusariosis de la espiga",
        "crops": "Trigo, cebada, maiz",
        "region": "Pampa humeda, NOA",
        "impact": "Micotoxinas en granos, rechazo de exportacion",
        "type": "Hongo",
    },
    "Xanthomonas citri": {
        "common": "Cancrosis de los citricos",
        "crops": "Naranjas, limones, pomelos",
        "region": "Tucuman, NEA, NOA",
        "impact": "Barrera fitosanitaria para exportacion",
        "type": "Bacteria Gram-negativa",
    },
    "Ralstonia solanacearum": {
        "common": "Marchitez bacteriana",
        "crops": "Tomate, papa, pimiento",
        "region": "Cinturon horticola",
        "impact": "Hasta 100% de perdida sin control",
        "type": "Bacteria Gram-negativa",
    },
}


def render_sidebar():
    """Render sidebar with controls."""
    with st.sidebar:
        st.markdown("### Configuracion del Analisis")

        pathogen = st.selectbox(
            "Patogeno objetivo",
            list(PATHOGEN_INFO.keys()),
            format_func=lambda x: f"{x} ({PATHOGEN_INFO[x]['common']})",
        )

        info = PATHOGEN_INFO[pathogen]
        st.markdown(f"""
        **Tipo:** {info['type']}
        **Cultivos:** {info['crops']}
        **Region:** {info['region']}
        **Impacto:** {info['impact']}
        """)

        st.divider()

        if not IS_CLOUD:
            with st.expander("Parametros avanzados", expanded=False):
                max_variants = st.slider("Max variantes por semilla", 50, 300, 150, 25)
                tox_threshold = st.slider("Umbral toxicidad", 0.1, 0.8, 0.4, 0.05)
        else:
            max_variants = 150
            tox_threshold = 0.4

        st.divider()

        precomputed_path = os.path.join(
            PRECOMPUTED_DIR, f"{pathogen.replace(' ', '_').lower()}.json"
        )

        if IS_CLOUD:
            # Cloud mode: auto-load precomputed, no pipeline button
            run_button = False
            load_precomputed = os.path.exists(precomputed_path)
            st.info("Modo cloud — resultados pre-computados. Para pipeline en vivo, correr localmente con GPU.")
        else:
            run_button = st.button(
                "Ejecutar Pipeline AgriAMP",
                type="primary",
                use_container_width=True,
            )
            load_precomputed = False
            if os.path.exists(precomputed_path):
                load_precomputed = st.checkbox("Cargar resultados pre-computados", value=False)

        st.divider()
        with st.expander("Que son los peptidos antimicrobianos (AMPs)?", expanded=False):
            st.markdown("""
            Los **AMPs** son cadenas cortas de aminoacidos (10-50) presentes
            en el **sistema inmune innato** de todos los organismos vivos.
            Llevan **3+ mil millones de anos** de evolucion combatiendo patogenos.

            **Mecanismo en 3 pasos:**
            1. **Atraccion electrostatica** — AMPs cationicos (+) se unen a la membrana microbiana anionica (-)
            2. **Insercion** — la cara hidrofobica penetra la bicapa lipidica
            3. **Disrupcion** — forman poros que destruyen la membrana

            **Ventaja vs pesticidas quimicos:** los pesticidas atacan UNA via
            metabolica y generan resistencia. Los AMPs atacan la **estructura
            fundamental de la membrana** — es la diferencia entre forzar una
            cerradura y tirar la puerta abajo.
            """)
        with st.expander("Como funciona AgriAMP", expanded=False):
            st.markdown("""
            **6 herramientas bioinformaticas** ejecutadas como workflow agentico:

            1. **Consulta DB** — 4,600+ AMPs de modlAMP + 27 curados antifungicos
            2. **Generador** — variantes optimizadas (K/R/L/W, carga, truncamiento)
            3. **ESM-2 Embeddings** — protein language model, 650M params, 1280-dim
            4. **Propiedades** — 12 descriptores bioquimicos (carga, GRAVY, pI, MW...)
            5. **Clasificador ML** — Random Forest, AUC 0.977, 5-fold CV estratificado
            6. **Toxicidad** — screening rule-based (GRAVY, carga, largo, Cys, WxxW)

            **Score:** `0.35*AMP + 0.25*carga + 0.20*anfipacidad + 0.10*estabilidad + 0.10*(1-tox)`
            """)
        st.divider()
        st.markdown("""
        **AgriAMP** v1.0
        Aleph Hackathon M26 — Track Biotech
        [GitHub](https://github.com/waitdeadai/agriamp)
        """)

        return pathogen, run_button, load_precomputed, precomputed_path, max_variants, tox_threshold



def render_top_candidates(df: pd.DataFrame):
    """Tab 1: Top candidates dashboard."""
    if df.empty:
        st.warning("No hay candidatos para mostrar.")
        return

    # Filter to passed toxicity
    passed = df[df["passed_toxicity"]].head(20)
    all_ranked = df.head(30)
    display_df = passed if len(passed) >= 5 else all_ranked

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Candidatos analizados", len(df))
    with col2:
        st.metric("Pasaron toxicidad", len(df[df["passed_toxicity"]]))
    with col3:
        st.metric("Top AgriAMP Score", f"{df['agriamp_score'].max():.3f}")
    with col4:
        avg_charge = display_df["net_charge"].mean()
        st.metric("Carga promedio", f"+{avg_charge:.1f}")

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
        labels={"agriamp_score": "AgriAMP Score", "label": "Peptido", "toxicity_risk": "Riesgo Tox."},
        title="Top 10 Candidatos por AgriAMP Score",
    )
    fig.add_hline(y=0.7, line_dash="dash", line_color="#4caf50", opacity=0.5,
                  annotation_text="Umbral candidato prometedor (0.70)")
    fig.update_layout(height=400, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("Tabla de Candidatos")
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
    if df.empty:
        st.warning("No hay datos para analizar.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Radar: Top Candidato vs Perfil AMP Ideal")

        # Ideal AMP profile (normalized 0-1)
        ideal = {
            "Carga": 0.7,
            "Anfipacidad": 0.8,
            "Estabilidad": 0.7,
            "Selectividad": 0.8,
            "Baja toxicidad": 0.9,
        }

        top = df.iloc[0]
        candidate = {
            "Carga": min(max(top["net_charge"] / 6, 0), 1),
            "Anfipacidad": min(max(top["hydrophobic_moment"] / 0.8, 0), 1),
            "Estabilidad": max(1 - top["instability_index"] / 100, 0),
            "Selectividad": top.get("selectivity_score", 0.5),
            "Baja toxicidad": 1 - top["toxicity_risk"],
        }

        categories = list(ideal.keys())

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(ideal.values()) + [list(ideal.values())[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Perfil AMP Ideal",
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
        st.subheader("Feature Importance del Clasificador")
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
                title="Top 10 Features Predictivas",
            )
            fig2.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Feature importances no disponibles (scoring por propiedades).")

    # Distribution plots
    st.subheader("Distribuciones de Propiedades")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.histogram(df, x="net_charge", nbins=30, title="Carga Neta (pH 7)",
                           color_discrete_sequence=["#2d8f2d"])
        fig.add_vline(x=2, line_dash="dash", line_color="red", annotation_text="Min ideal (+2)")
        fig.add_vline(x=9, line_dash="dash", line_color="red", annotation_text="Max ideal (+9)")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="hydrophobic_moment", nbins=30, title="Momento Hidrofobico",
                           color_discrete_sequence=["#1a5e1a"])
        fig.add_vline(x=0.3, line_dash="dash", line_color="red", annotation_text="Umbral anfipacidad")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = px.histogram(df, x="molecular_weight", nbins=30, title="Peso Molecular (Da)",
                           color_discrete_sequence=["#4caf50"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def render_sequence_viewer(df: pd.DataFrame):
    """Tab 3: Interactive sequence viewer with colored amino acids."""
    if df.empty:
        st.warning("No hay secuencias para visualizar.")
        return

    top_candidates = df.head(10)
    selected_idx = st.selectbox(
        "Seleccionar peptido",
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

    st.markdown("### Secuencia coloreada")
    st.markdown(f"<div style='line-height:2.5;'>{html_seq}</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px;font-size:12px;">
    <span style="background-color:#FFCA28;padding:2px 6px;border-radius:3px;">Hidrofobico</span>
    <span style="background-color:#42A5F5;padding:2px 6px;border-radius:3px;color:white;">Cationico (+)</span>
    <span style="background-color:#E57373;padding:2px 6px;border-radius:3px;">Anionico (-)</span>
    <span style="background-color:#66BB6A;padding:2px 6px;border-radius:3px;">Polar</span>
    </div>
    """, unsafe_allow_html=True)

    # Property cards
    st.markdown("### Propiedades")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Carga neta", f"+{pep['net_charge']:.1f}")
        st.metric("Peso molecular", f"{pep['molecular_weight']:.0f} Da")
    with col2:
        st.metric("Momento hidrofobico", f"{pep['hydrophobic_moment']:.3f}")
        st.metric("GRAVY", f"{pep['gravy']:.3f}")
    with col3:
        st.metric("Punto isoelectrico", f"{pep['isoelectric_point']:.1f}")
        st.metric("Indice Boman", f"{pep['boman_index']:.2f}")
    with col4:
        st.metric("Toxicidad", f"{pep['toxicity_risk']:.2f}")
        st.metric("AgriAMP Score", f"{pep['agriamp_score']:.4f}")

    # Score breakdown
    st.markdown("### Composicion del AgriAMP Score")
    _charge_norm = min(max(pep["net_charge"] / 6, 0), 1)
    _hm_norm = min(max(pep["hydrophobic_moment"] / 0.8, 0), 1)
    _stab_norm = max(1 - pep["instability_index"] / 100, 0)
    _tox_comp = 1 - pep["toxicity_risk"]
    _components = {
        "AMP Probability (35%)": pep["amp_probability"] * 0.35,
        "Carga neta (25%)": _charge_norm * 0.25,
        "Anfipacidad (20%)": _hm_norm * 0.20,
        "Estabilidad (10%)": _stab_norm * 0.10,
        "Baja toxicidad (10%)": _tox_comp * 0.10,
    }
    _comp_df = pd.DataFrame({"Componente": list(_components.keys()), "Contribucion": list(_components.values())})
    _fig_comp = px.bar(_comp_df, x="Contribucion", y="Componente", orientation="h",
                       color="Contribucion", color_continuous_scale="Greens",
                       title=f"Score total: {pep['agriamp_score']:.4f}")
    _fig_comp.update_layout(height=250, showlegend=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(_fig_comp, use_container_width=True)

    # Helical wheel projection
    st.markdown("### Proyeccion Helical Wheel")
    st.caption("Muestra la distribucion anfipática — residuos hidrofobicos (amarillo) vs cationicos (azul) en una helice alfa")

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
    if not metrics:
        st.warning("No hay metricas de validacion.")
        return

    cv_auc = metrics.get("cv_auc_mean")

    if cv_auc is None:
        st.info("Se uso scoring basado en propiedades (no ML). Metricas de validacion cruzada no disponibles.")
        return

    # Comprehensive metrics row
    st.subheader("Metricas del Clasificador (Out-of-Fold)")
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
        st.metric("Sensibilidad", f"{metrics.get('oof_sensitivity', 0):.3f}")
    with col_m6:
        st.metric("Especificidad", f"{metrics.get('oof_specificity', 0):.3f}")

    st.caption(
        f"Datos: {metrics.get('n_train_positive', 0)} AMPs + "
        f"{metrics.get('n_train_negative', 0)} non-AMPs | "
        f"5-fold CV estratificado | AUC std: ±{metrics.get('cv_auc_std', 0):.3f}"
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
                title="Matriz de Confusion (Out-of-Fold)",
                height=350,
                xaxis_title="Prediccion",
                yaxis_title="Real",
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
                title="AUC-ROC por Fold de Cross-Validation",
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
                title="Curva ROC",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_caso_real(df: pd.DataFrame, pathogen: str):
    """Tab: Caso Real — real case study with verified data."""
    st.markdown("""
    <div class="hero-box">
        <h3>Caso Real: Botrytis cinerea en Mendoza</h3>
        <p style="font-size:1.05rem;">
            La podredumbre gris (<i>Botrytis cinerea</i>) es el hongo mas agresivo y
            prevalente en los vinedos de Mendoza. La literatura reporta perdidas de
            hasta <b>50-80% de la cosecha</b> en anos de infeccion severa, con impacto
            significativo en rendimiento y calidad del fruto (PMC, Springer).
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Perdida por Botrytis", "50-80%", delta=None, help="Rango en anos severos (PMC, Springer)")
        st.metric("Exportaciones vino ARG", "~$800M/ano", help="Mendoza = 75% produccion nacional")
    with col2:
        st.metric("Perdidas globales/ano", "~$2B USD", delta=None, help="Impacto economico global de B. cinerea")
        st.metric("Agroquimicos ARG/ano", "500M+ litros/kg", help="SPRINT-H2020, Mongabay 2020")
    with col3:
        st.metric("Perdida en peso", "53%", delta=None, help="Peso de cosecha perdido")
        st.metric("Pesticidas ARG banned en UE", "106", help="Acuerdo EU-Mercosur, enero 2026")

    st.divider()

    st.markdown("#### Crisis Regulatoria")
    st.markdown("""
    - **Nuevos limites MRL de importacion UE** (marzo 2026): residuos de clothianidin y thiamethoxam prohibidos en productos importados (ban de uso outdoor desde 2018)
    - **Acuerdo EU-Mercosur**: Argentina tiene **106 principios activos prohibidos** en la UE — barrera para exportacion de vinos
    - **Resistencia documentada**: mutaciones duales SDHI + QoI en poblaciones de *B. cinerea* a nivel global
    - **Biopesticidas basados en peptidos aprobados para vid**: **ninguno** — oportunidad de mercado clara
    """)

    st.divider()

    st.markdown("#### Validacion con AMPs Publicados")
    st.markdown("""
    | AMP | MIC vs *B. cinerea* | Fuente | Referencia |
    |-----|---------------------|--------|------------|
    | **Epinecidin-1** | 12.5 umol/L | *Epinephelus coioides* (mero) | Food Chemistry, 2022 |
    | **EPI-4** (variante optimizada) | **6 umol/L** (2x mejor) | Sintetico | ACS JAFC, 2025 |
    | **Rs-AFP2** | 3 umol/L | *Raphanus sativus* (rabano) | Plant defensin literature |
    | **NaD1** | 2 umol/L | *Nicotiana alata* (tabaco) | Plant defensin literature |

    EPI-4 es una variante optimizada de Epinecidin-1 con sustituciones en la superficie polar
    (Pan et al., JAFC 2025). Nuestro generador de variantes aplica estrategias similares:
    sustitucion con residuos cationicos (+K/R) para aumentar carga y actividad antimicrobiana.
    """)

    # Show Epinecidin in results if present
    if not df.empty:
        epi_matches = df[df["name"].str.contains("Epinecidin|EPI-4", case=False, na=False)]
        if not epi_matches.empty:
            st.divider()
            st.markdown("#### Epinecidin en nuestros resultados")
            for _, row in epi_matches.iterrows():
                st.markdown(
                    f"- **{row['name']}**: AgriAMP Score **{row['agriamp_score']:.4f}** | "
                    f"Carga {row['net_charge']:+.1f} | AMP prob {row['amp_probability']:.3f} | "
                    f"Toxicidad {row['toxicity_risk']:.2f} {'(aprobado)' if row['passed_toxicity'] else '(flaggeado)'}"
                )

    st.divider()
    st.markdown("#### Proximo Paso: Validacion Experimental")
    st.markdown("""
    Los candidatos de AgriAMP son **candidatos in silico para validacion experimental**.
    El siguiente paso es **sintesis quimica** ($80-250/peptido screening) y ensayos de
    inhibicion in vitro contra *B. cinerea*. Socio natural: **INTA Mendoza** (Laboratorio
    de Fitopatologia, equipo de Dra. Georgina Escoriaza).
    """)


def render_benchmark(metrics: dict):
    """Tab: Benchmark comparison vs published SOTA tools."""
    st.markdown("### AgriAMP vs Estado del Arte (2024-2026)")
    st.markdown("""
    Comparacion de metricas contra herramientas publicadas de prediccion de AMPs.
    Todas las metricas de AgriAMP son **out-of-fold** (validacion cruzada estratificada 5-fold,
    sin data leakage).
    """)

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
            "Herramienta": "AgriAMP (nuestro)",
            "Ano": "2026",
            "Metodo": "ESM-2 + RF",
            "Dataset": f"{n_train} seqs",
            "AUC": our_auc,
            "MCC": our_mcc,
            "Sensibilidad": our_sens,
            "Especificidad": our_spec,
            "Ref": "Este trabajo",
        },
        {
            "Herramienta": "amPEPpy",
            "Ano": "2020",
            "Metodo": "RF + features",
            "Dataset": "Custom",
            "AUC": 0.99,
            "MCC": 0.90,
            "Sensibilidad": None,
            "Especificidad": None,
            "Ref": "Bioinformatics, 2021",
        },
        {
            "Herramienta": "PLAPD",
            "Ano": "2025",
            "Metodo": "ESM-2 + CNN + Transformer",
            "Dataset": "8,268 seqs",
            "AUC": 0.922,
            "MCC": 0.749,
            "Sensibilidad": None,
            "Especificidad": 0.946,
            "Ref": "Methods, 2025",
        },
        {
            "Herramienta": "AMP-RNNpro",
            "Ano": "2024",
            "Metodo": "RNN + prob. features",
            "Dataset": "Custom",
            "AUC": None,
            "MCC": None,
            "Sensibilidad": 0.965,
            "Especificidad": 0.979,
            "Ref": "Sci. Reports, 2024",
        },
        {
            "Herramienta": "sAMPpred-GAT",
            "Ano": "2023",
            "Metodo": "GAT + estructura",
            "Dataset": "Multiple",
            "AUC": None,
            "MCC": None,
            "Sensibilidad": None,
            "Especificidad": None,
            "Ref": "8 test sets, 2023",
        },
    ]

    bench_df = pd.DataFrame(benchmark_data)

    def fmt_metric(v):
        if v is None:
            return "—"
        return f"{v:.3f}"

    display_df = bench_df.copy()
    for col in ["AUC", "MCC", "Sensibilidad", "Especificidad"]:
        display_df[col] = display_df[col].apply(fmt_metric)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption("""
    *Nota: Las metricas NO son directamente comparables entre herramientas (cada una usa datasets diferentes).
    amPEPpy reporta AUC 0.99 en su propio dataset custom. AgriAMP usa el benchmark estandar modlAMP.
    La comparacion muestra que nuestro pipeline es competitivo en el rango SOTA.*
    """)

    # AUC comparison bar chart
    auc_tools = [d for d in benchmark_data if d["AUC"] is not None]
    if auc_tools:
        fig = go.Figure()
        colors = ["#2d8f2d" if d["Herramienta"].startswith("AgriAMP") else "#666"
                  for d in auc_tools]
        fig.add_trace(go.Bar(
            x=[d["Herramienta"] for d in auc_tools],
            y=[d["AUC"] for d in auc_tools],
            marker_color=colors,
            text=[f"{d['AUC']:.3f}" for d in auc_tools],
            textposition="outside",
        ))
        fig.update_layout(
            title="AUC-ROC: AgriAMP vs Herramientas Publicadas",
            yaxis_title="AUC-ROC",
            yaxis=dict(range=[0.85, 1.02]),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("#### Diferenciador de AgriAMP")
    st.markdown("""
    Las herramientas comparadas son **clasificadores**: reciben una secuencia y predicen
    si es AMP. AgriAMP es un **pipeline agentico completo** que:

    1. Consulta bases de datos de AMPs conocidos para el patogeno especifico
    2. Genera variantes optimizadas con mutaciones dirigidas
    3. Genera embeddings con ESM-2 (650M params)
    4. Calcula 12 propiedades bioquimicas
    5. Clasifica con Random Forest (AUC competitivo)
    6. Filtra por toxicidad y selectividad

    **Ninguna otra herramienta ofrece este workflow integrado y accesible para agronomos.**
    """)

    # Our full metrics summary
    st.divider()
    st.markdown("#### Metricas Completas de AgriAMP")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AUC-ROC", f"{our_auc:.3f}")
        st.metric("MCC", f"{our_mcc:.3f}")
    with col2:
        st.metric("Accuracy", f"{our_acc:.3f}")
        st.metric("F1-Score", f"{our_f1:.3f}")
    with col3:
        st.metric("Sensibilidad", f"{our_sens:.3f}")
        st.metric("Especificidad", f"{our_spec:.3f}")
    with col4:
        st.metric("Datos entrenamiento", f"{n_train}")
        st.metric("CV Folds", "5 (estratificado)")


def render_export(df: pd.DataFrame, pathogen: str):
    """Tab 5: Export results."""
    if df.empty:
        st.warning("No hay resultados para exportar.")
        return

    st.subheader("Exportar Resultados")

    col1, col2 = st.columns(2)

    with col1:
        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            "Descargar CSV completo",
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
            "Descargar FASTA (top candidatos)",
            data=fasta_text,
            file_name=f"agriamp_{pathogen.replace(' ', '_').lower()}_top.fasta",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Resumen del Analisis")
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
        st.markdown('<p class="subtitle">Pipeline Agentico de IA para Descubrimiento de Peptidos Antimicrobianos contra Patogenos de Cultivos</p>', unsafe_allow_html=True)

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
            st.success("Resultados pre-computados cargados.")

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
            st.subheader("Workflow Agentico")
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
                                st.caption(f"Duracion: {s.duration:.1f}s")

            with st.spinner("Ejecutando pipeline AgriAMP..."):
                result = agent.run(pathogen=pathogen, callback=on_step)

            if result.success:
                st.session_state.agent_result = {
                    "candidates": result.candidates,
                    "metrics": result.metrics,
                    "steps": [(s.tool_name, s.icon, s.status, s.message, s.duration) for s in result.steps],
                    "total_duration": result.total_duration,
                }
                st.success(f"Pipeline completado en {result.total_duration:.1f}s — {len(result.candidates)} candidatos analizados")

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
                st.error("El pipeline encontro errores. Revisa el log de arriba.")

    # Display results
    if st.session_state.agent_result is not None:
        data = st.session_state.agent_result
        df = data["candidates"] if isinstance(data["candidates"], pd.DataFrame) else pd.DataFrame(data["candidates"])
        metrics = data.get("metrics", {})

        # Show agent workflow log
        steps = data.get("steps", [])
        if steps:
            with st.expander("Workflow Agentico — 6 pasos completados", expanded=False):
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
            st.caption(f"Pipeline completado en {_total_dur:.1f}s — {_n_cands} candidatos analizados, {_n_passed} pasaron screening de toxicidad")

        _pinfo = PATHOGEN_INFO[pathogen]
        st.markdown(f"### Resultados para *{pathogen}* ({_pinfo['common']})")
        st.caption(f"{_pinfo['type']} | Cultivos: {_pinfo['crops']} | {_pinfo['impact']}")

        st.divider()

        # Results tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🌍 Caso Real",
            "🏆 Top Candidatos",
            "📊 Analisis de Propiedades",
            "🧬 Visor de Secuencias",
            "📈 Benchmark vs SOTA",
            "✅ Validacion ML",
            "📥 Exportar",
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
        st.markdown("""
        <div class="hero-box">
            <h3>De tu cultivo a la solucion biologica</h3>
            <p style="font-size:1.05rem;margin-bottom:16px;">
                AgriAMP es un agente de bioinformatica que disena <b>peptidos antimicrobianos</b>
                (biopesticidas naturales) contra patogenos de cultivos usando inteligencia artificial.
                Describe tu problema — el agente ejecuta un pipeline completo de 6 pasos y
                entrega candidatos peptidicos rankeados para sintesis.
            </p>
            <div style="display:flex;gap:40px;flex-wrap:wrap;">
                <div><div class="hero-stat">650M</div><div class="hero-label">parametros ESM-2</div></div>
                <div><div class="hero-stat">0.977</div><div class="hero-label">AUC clasificador</div></div>
                <div><div class="hero-stat">4,600+</div><div class="hero-label">secuencias entrenamiento</div></div>
                <div><div class="hero-stat">6</div><div class="hero-label">tools bioinformaticos</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            #### Problema
            Argentina usa **500M+ litros/kg de agroquimicos/ano**. Los patogenos generan resistencia.
            La UE endurece bans. Los agronomos no tienen acceso a bioinformatica.
            """)
        with col2:
            st.markdown("""
            #### Solucion
            **Peptidos antimicrobianos** (AMPs) — armas del sistema inmune innato con
            3+ mil millones de anos de evolucion. Atacan la membrana, no una via metabolica.
            """)
        with col3:
            st.markdown("""
            #### Pipeline Agentico
            **6 herramientas** orquestadas como workflow de IA:
            consulta DB → embeddings ESM-2 → propiedades → clasificador ML → generador → toxicidad.
            """)

        st.info("Selecciona un patogeno en la barra lateral y presiona **Ejecutar Pipeline AgriAMP** para comenzar.")


if __name__ == "__main__":
    main()
