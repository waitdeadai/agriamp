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
import time

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
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a5e1a, #2d8f2d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-top: -10px;
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

        with st.expander("Parametros avanzados", expanded=False):
            max_variants = st.slider("Max variantes por semilla", 50, 300, 150, 25)
            tox_threshold = st.slider("Umbral toxicidad", 0.1, 0.8, 0.4, 0.05)

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
        st.markdown("""
        **AgriAMP** v1.0
        Pipeline agentico de bioinformatica para descubrimiento
        de peptidos antimicrobianos contra patogenos de cultivos.

        Desarrollado para Aleph Hackathon M26 — Track Biotech

        [GitHub](https://github.com/waitdeadai/agriamp)
        """)

        return pathogen, run_button, load_precomputed, precomputed_path, max_variants, tox_threshold


def render_agent_log(steps):
    """Render the agentic workflow log."""
    for step in steps:
        icon = {"success": "✅", "warning": "⚠️", "error": "❌", "running": "⏳"}.get(step.status, "⏳")
        with st.expander(f"{step.icon} {step.tool_name} {icon}", expanded=(step.status != "success")):
            st.markdown(step.message)
            if step.duration > 0:
                st.caption(f"Duracion: {step.duration:.1f}s")


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
    fig.update_layout(height=400)
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
            "Carga": min(max(top["net_charge"] / 8, 0), 1),
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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance del Clasificador")
        st.metric("AUC-ROC (Cross-validation)", f"{cv_auc:.3f}")
        st.metric("AUC-ROC (Training)", f"{metrics.get('train_auc', 0):.3f}")
        st.metric("Positivos de entrenamiento", metrics.get("n_train_positive", 0))
        st.metric("Negativos de entrenamiento", metrics.get("n_train_negative", 0))

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
        st.subheader("Curva ROC")
        y_true = metrics.get("y_train_true", [])
        y_pred = metrics.get("y_train_pred", [])

        if y_true and y_pred:
            from sklearn.metrics import roc_curve, auc

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})",
                                     line=dict(color="#2d8f2d", width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                     line=dict(color="gray", dash="dash")))
            fig.update_layout(
                title="Curva ROC (Training Set)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


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
    # Header
    st.markdown('<p class="main-header">AgriAMP</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Pipeline Agentico de IA para Descubrimiento de Peptidos Antimicrobianos contra Patogenos de Cultivos</p>', unsafe_allow_html=True)

    # Sidebar
    pathogen, run_button, load_precomputed, precomputed_path, max_variants, tox_threshold = render_sidebar()

    # Initialize session state
    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None
    if "agent_steps" not in st.session_state:
        st.session_state.agent_steps = []

    # Load precomputed if requested
    if load_precomputed and os.path.exists(precomputed_path):
        with open(precomputed_path, "r") as f:
            cached = json.load(f)
        st.session_state.agent_result = {
            "candidates": pd.DataFrame(cached["candidates"]),
            "metrics": cached.get("metrics", {}),
            "steps": cached.get("steps", []),
            "total_duration": cached.get("total_duration", 0),
        }
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
            with st.expander("Workflow Agentico (6 pasos)", expanded=False):
                for step in steps:
                    if isinstance(step, (list, tuple)) and len(step) >= 4:
                        name, icon, status, msg = step[0], step[1], step[2], step[3]
                        dur = step[4] if len(step) > 4 else 0
                    else:
                        continue
                    status_icon = {"success": "OK", "warning": "WARN", "error": "ERR"}.get(status, "?")
                    st.markdown(f"**{icon} {name}** [{status_icon}] — {msg}")
                    if dur and dur > 0:
                        st.caption(f"Duracion: {dur:.1f}s")

            st.caption(f"Pipeline total: {data.get('total_duration', 0):.1f}s")

        st.divider()

        # Results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Top Candidatos",
            "📊 Analisis de Propiedades",
            "🧬 Visor de Secuencias",
            "✅ Validacion ML",
            "📥 Exportar",
        ])

        with tab1:
            render_top_candidates(df)
        with tab2:
            render_property_analysis(df, metrics)
        with tab3:
            render_sequence_viewer(df)
        with tab4:
            render_validation(metrics)
        with tab5:
            render_export(df, pathogen)

    else:
        # Welcome screen
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 🔬 Problema
            Los agroquimicos contaminan suelos, generan resistencia y enfrentan
            restricciones regulatorias crecientes. Argentina usa 500M litros/ano.
            """)
        with col2:
            st.markdown("""
            ### 🧬 Solucion
            AgriAMP usa IA (ESM-2 protein language model + ML) para disenar
            peptidos antimicrobianos como biopesticidas alternativos.
            """)
        with col3:
            st.markdown("""
            ### 🚀 Pipeline
            6 herramientas bioinformaticas orquestadas como workflow agentico:
            consulta DB, embeddings, propiedades, clasificacion, generacion, toxicidad.
            """)

        st.info("Selecciona un patogeno y presiona **Ejecutar Pipeline AgriAMP** para comenzar el analisis.")


if __name__ == "__main__":
    main()
