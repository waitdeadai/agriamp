"""Bilingual strings for AgriAMP (EN/ES). Default: English."""

PATHOGEN_INFO = {
    "en": {
        "Botrytis cinerea": {
            "common": "Gray mold",
            "crops": "Grapes, strawberries, tomatoes",
            "region": "Mendoza, Patagonia",
            "impact": "Up to 30% loss in humid seasons",
            "type": "Fungus",
        },
        "Fusarium graminearum": {
            "common": "Fusarium head blight",
            "crops": "Wheat, barley, corn",
            "region": "Humid Pampas, NOA",
            "impact": "Mycotoxins in grains, export rejection",
            "type": "Fungus",
        },
        "Xanthomonas citri": {
            "common": "Citrus canker",
            "crops": "Oranges, lemons, grapefruit",
            "region": "Tucuman, NEA, NOA",
            "impact": "Phytosanitary barrier for export",
            "type": "Gram-negative bacterium",
        },
        "Ralstonia solanacearum": {
            "common": "Bacterial wilt",
            "crops": "Tomato, potato, pepper",
            "region": "Horticultural belt",
            "impact": "Up to 100% loss without control",
            "type": "Gram-negative bacterium",
        },
    },
    "es": {
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
    },
}

STRINGS = {
    "en": {
        # ── Header ──
        "subtitle": "AI-Powered Agentic Pipeline for Antimicrobial Peptide Discovery Against Crop Pathogens",

        # ── Sidebar ──
        "analysis_config": "Analysis Configuration",
        "target_pathogen": "Target Pathogen",
        "type": "Type",
        "crops": "Crops",
        "region": "Region",
        "impact": "Impact",
        "advanced_params": "Advanced Parameters",
        "max_variants": "Max variants per seed",
        "tox_threshold": "Toxicity threshold",
        "cloud_mode": "Cloud mode — precomputed results. For live pipeline, run locally with GPU.",
        "run_pipeline": "Run AgriAMP Pipeline",
        "load_precomputed": "Load precomputed results",
        "what_are_amps": "What are antimicrobial peptides (AMPs)?",
        "amps_explanation": """
**AMPs** are short amino acid chains (10-50) found in the
**innate immune system** of all living organisms.
They've been **3+ billion years** in evolution fighting pathogens.

**Mechanism in 3 steps:**
1. **Electrostatic attraction** — cationic AMPs (+) bind to anionic microbial membranes (-)
2. **Insertion** — the hydrophobic face penetrates the lipid bilayer
3. **Disruption** — they form pores that destroy the membrane

**Advantage vs chemical pesticides:** pesticides attack ONE metabolic
pathway and generate resistance. AMPs attack the **fundamental
structure of the membrane** — it's the difference between picking
a lock and kicking the door down.
""",
        "how_agriamp_works": "How AgriAMP works",
        "agriamp_explanation": """
**6 bioinformatics tools** executed as an agentic workflow:

1. **DB Query** — 4,600+ AMPs from modlAMP + 27 curated antifungals
2. **Generator** — optimized variants (K/R/L/W, charge, truncation)
3. **ESM-2 Embeddings** — protein language model, 650M params, 1280-dim
4. **Properties** — 12 biochemical descriptors (charge, GRAVY, pI, MW...)
5. **ML Classifier** — Random Forest, AUC 0.977, 5-fold stratified CV
6. **Toxicity** — rule-based screening (GRAVY, charge, length, Cys, WxxW)

**Score:** `0.35*AMP + 0.25*charge + 0.20*amphipathicity + 0.10*stability + 0.10*(1-tox)`
""",

        # ── Welcome Screen ──
        "hero_title": "From your crop to the biological solution",
        "hero_description": """
AgriAMP is a bioinformatics agent that designs <b>antimicrobial peptides</b>
(natural biopesticides) against crop pathogens using artificial intelligence.
Describe your problem — the agent runs a complete 6-step pipeline and
delivers ranked peptide candidates ready for synthesis.
""",
        "hero_esm2": "ESM-2 parameters",
        "hero_auc": "AUC classifier",
        "hero_seqs": "training sequences",
        "hero_tools": "bioinformatics tools",
        "problem": "Problem",
        "problem_desc": """
Argentina uses **500M+ liters/kg of agrochemicals/year**. Pathogens develop resistance.
The EU tightens bans. Agronomists lack access to bioinformatics tools.
""",
        "solution": "Solution",
        "solution_desc": """
**Antimicrobial peptides** (AMPs) — weapons of the innate immune system with
3+ billion years of evolution. They attack the membrane, not a single metabolic pathway.
""",
        "agentic_pipeline": "Agentic Pipeline",
        "pipeline_desc": """
**6 tools** orchestrated as an AI workflow:
DB query → ESM-2 embeddings → properties → ML classifier → generator → toxicity.
""",
        "select_pathogen_info": "Select a pathogen in the sidebar and press **Run AgriAMP Pipeline** to begin.",

        # ── Tab Names ──
        "tab_real_case": "Real Case",
        "tab_top_candidates": "Top Candidates",
        "tab_property_analysis": "Property Analysis",
        "tab_sequence_viewer": "Sequence Viewer",
        "tab_benchmark": "Benchmark vs SOTA",
        "tab_ml_validation": "ML Validation",
        "tab_export": "Export",

        # ── Pipeline Execution ──
        "agentic_workflow": "Agentic Workflow",
        "duration": "Duration",
        "running_pipeline": "Running AgriAMP pipeline...",
        "pipeline_completed": "Pipeline completed in {dur:.1f}s — {n} candidates analyzed",
        "pipeline_error": "The pipeline encountered errors. Check the log above.",
        "precomputed_loaded": "Precomputed results loaded.",
        "workflow_steps_completed": "Agentic Workflow — 6 steps completed",
        "pipeline_summary": "Pipeline completed in {dur:.1f}s — {n_cands} candidates analyzed, {n_passed} passed toxicity screening",
        "results_for": "Results for *{pathogen}* ({common})",

        # ── Tab: Top Candidates ──
        "no_candidates": "No candidates to display.",
        "candidates_analyzed": "Candidates analyzed",
        "passed_toxicity": "Passed toxicity",
        "avg_charge": "Average charge",
        "peptide": "Peptide",
        "tox_risk": "Tox. Risk",
        "top10_title": "Top 10 Candidates by AgriAMP Score",
        "threshold_line": "Promising candidate threshold (0.70)",
        "candidate_table": "Candidate Table",

        # ── Tab: Property Analysis ──
        "no_data": "No data to analyze.",
        "radar_title": "Radar: Top Candidate vs Ideal AMP Profile",
        "charge": "Charge",
        "amphipathicity": "Amphipathicity",
        "stability": "Stability",
        "selectivity": "Selectivity",
        "low_toxicity": "Low toxicity",
        "ideal_amp_profile": "Ideal AMP Profile",
        "feature_importance": "Classifier Feature Importance",
        "top10_features": "Top 10 Predictive Features",
        "no_feature_importance": "Feature importances not available (property-based scoring).",
        "property_distributions": "Property Distributions",
        "net_charge_title": "Net Charge (pH 7)",
        "min_ideal": "Min ideal (+2)",
        "max_ideal": "Max ideal (+9)",
        "hydrophobic_moment_title": "Hydrophobic Moment",
        "amphipathicity_threshold": "Amphipathicity threshold",
        "molecular_weight_title": "Molecular Weight (Da)",

        # ── Tab: Sequence Viewer ──
        "no_sequences": "No sequences to display.",
        "select_peptide": "Select peptide",
        "colored_sequence": "Colored Sequence",
        "hydrophobic": "Hydrophobic",
        "cationic": "Cationic (+)",
        "anionic": "Anionic (-)",
        "polar": "Polar",
        "properties": "Properties",
        "net_charge": "Net charge",
        "molecular_weight": "Molecular weight",
        "hydrophobic_moment": "Hydrophobic moment",
        "isoelectric_point": "Isoelectric point",
        "boman_index": "Boman index",
        "toxicity": "Toxicity",
        "score_composition": "AgriAMP Score Composition",
        "amp_prob_component": "AMP Probability (35%)",
        "charge_component": "Net charge (25%)",
        "amphipathicity_component": "Amphipathicity (20%)",
        "stability_component": "Stability (10%)",
        "low_tox_component": "Low toxicity (10%)",
        "component": "Component",
        "contribution": "Contribution",
        "helical_wheel": "Helical Wheel Projection",
        "helical_caption": "Shows amphipathic distribution — hydrophobic residues (yellow) vs cationic (blue) in an alpha helix",
        "approved": "(approved)",
        "flagged": "(flagged)",

        # ── Tab: Benchmark ──
        "benchmark_title": "AgriAMP vs State of the Art (2024-2026)",
        "benchmark_description": """
Metric comparison against published AMP prediction tools.
All AgriAMP metrics are **out-of-fold** (5-fold stratified cross-validation,
no data leakage).
""",
        "tool": "Tool",
        "year": "Year",
        "method": "Method",
        "dataset": "Dataset",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "ref": "Ref",
        "agriamp_ours": "AgriAMP (ours)",
        "this_work": "This work",
        "benchmark_note": """
*Note: Metrics are NOT directly comparable across tools (each uses different datasets).
amPEPpy reports AUC 0.99 on its own custom dataset. AgriAMP uses the standard modlAMP benchmark.
The comparison shows our pipeline is competitive in the SOTA range.*
""",
        "auc_chart_title": "AUC-ROC: AgriAMP vs Published Tools",
        "differentiator_title": "AgriAMP Differentiator",
        "differentiator_content": """
The compared tools are **classifiers**: they receive a sequence and predict
whether it's an AMP. AgriAMP is a **complete agentic pipeline** that:

1. Queries known AMP databases for the specific pathogen
2. Generates optimized variants with directed mutations
3. Generates embeddings with ESM-2 (650M params)
4. Computes 12 biochemical properties
5. Classifies with Random Forest (competitive AUC)
6. Filters by toxicity and selectivity

**No other tool offers this integrated workflow accessible to agronomists.**
""",
        "full_metrics": "AgriAMP Full Metrics",
        "training_data": "Training data",
        "cv_folds": "CV Folds",
        "stratified": "5 (stratified)",

        # ── Tab: ML Validation ──
        "no_validation": "No validation metrics available.",
        "property_scoring_info": "Property-based scoring was used (no ML). Cross-validation metrics not available.",
        "classifier_metrics": "Classifier Metrics (Out-of-Fold)",
        "data_caption": "Data: {pos} AMPs + {neg} non-AMPs | 5-fold stratified CV | AUC std: \u00b1{std:.3f}",
        "confusion_matrix": "Confusion Matrix (Out-of-Fold)",
        "prediction": "Prediction",
        "actual": "Actual",
        "auc_per_fold": "AUC-ROC per Cross-Validation Fold",
        "roc_curve": "ROC Curve",

        # ── Tab: Export ──
        "no_export": "No results to export.",
        "export_results": "Export Results",
        "download_csv": "Download full CSV",
        "download_fasta": "Download FASTA (top candidates)",
        "analysis_summary": "Analysis Summary",

        # ── Tab: Real Case ──
        "caso_real_hero": """
<div class="hero-box">
    <h3>Real Case: Botrytis cinerea in Mendoza</h3>
    <p style="font-size:1.05rem;">
        Gray mold (<i>Botrytis cinerea</i>) is the most aggressive and
        prevalent fungus in Mendoza vineyards. Published literature reports losses of
        up to <b>50-80% of the harvest</b> in severe infection years, with significant
        impact on yield and fruit quality (PMC, Springer).
    </p>
</div>
""",
        "loss_botrytis": "Loss from Botrytis",
        "loss_botrytis_help": "Range in severe years (PMC, Springer)",
        "wine_exports": "Wine exports ARG",
        "wine_exports_help": "Mendoza = 75% national production",
        "global_losses": "Global losses/year",
        "global_losses_help": "Global economic impact of B. cinerea",
        "agrochemicals": "Agrochemicals ARG/year",
        "agrochemicals_help": "SPRINT-H2020, Mongabay 2020",
        "weight_loss": "Weight loss",
        "weight_loss_help": "Harvest weight lost",
        "banned_pesticides": "ARG pesticides banned in EU",
        "banned_pesticides_help": "EU-Mercosur agreement, Jan 2026",
        "regulatory_crisis": "Regulatory Crisis",
        "regulatory_content": """
- **New EU import MRL limits** (March 2026): clothianidin and thiamethoxam residues banned in imported products (outdoor use ban since 2018)
- **EU-Mercosur agreement**: Argentina has **106 banned active ingredients** in the EU — barrier for wine exports
- **Documented resistance**: dual SDHI + QoI mutations in *B. cinerea* populations globally
- **Peptide-based biopesticides approved for grapevines**: **none** — clear market opportunity
""",
        "validated_amps": "Validation with Published AMPs",
        "validated_amps_table": """
| AMP | MIC vs *B. cinerea* | Source | Reference |
|-----|---------------------|--------|-----------|
| **Epinecidin-1** | 12.5 umol/L | *Epinephelus coioides* (grouper) | Food Chemistry, 2022 |
| **EPI-4** (optimized variant) | **6 umol/L** (2x better) | Synthetic | ACS JAFC, 2025 |
| **Rs-AFP2** | 3 umol/L | *Raphanus sativus* (radish) | Plant defensin literature |
| **NaD1** | 2 umol/L | *Nicotiana alata* (tobacco) | Plant defensin literature |

EPI-4 is an optimized variant of Epinecidin-1 with substitutions on the polar surface
(Pan et al., JAFC 2025). Our variant generator applies similar strategies:
substitution with cationic residues (+K/R) to increase charge and antimicrobial activity.
""",
        "epinecidin_in_results": "Epinecidin in Our Results",
        "epinecidin_row": "- **{name}**: AgriAMP Score **{score:.4f}** | Charge {charge:+.1f} | AMP prob {prob:.3f} | Toxicity {tox:.2f} {status}",
        "next_step": "Next Step: Experimental Validation",
        "next_step_content": """
AgriAMP candidates are **in silico candidates for experimental validation**.
The next step is **chemical synthesis** ($80-250/peptide screening) and
in vitro inhibition assays against *B. cinerea*. Natural partner: **INTA Mendoza**
(Phytopathology Laboratory, Dr. Georgina Escoriaza's team).
""",
    },

    "es": {
        # ── Header ──
        "subtitle": "Pipeline Agentico de IA para Descubrimiento de Peptidos Antimicrobianos contra Patogenos de Cultivos",

        # ── Sidebar ──
        "analysis_config": "Configuracion del Analisis",
        "target_pathogen": "Patogeno objetivo",
        "type": "Tipo",
        "crops": "Cultivos",
        "region": "Region",
        "impact": "Impacto",
        "advanced_params": "Parametros avanzados",
        "max_variants": "Max variantes por semilla",
        "tox_threshold": "Umbral toxicidad",
        "cloud_mode": "Modo cloud — resultados pre-computados. Para pipeline en vivo, correr localmente con GPU.",
        "run_pipeline": "Ejecutar Pipeline AgriAMP",
        "load_precomputed": "Cargar resultados pre-computados",
        "what_are_amps": "Que son los peptidos antimicrobianos (AMPs)?",
        "amps_explanation": """
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
""",
        "how_agriamp_works": "Como funciona AgriAMP",
        "agriamp_explanation": """
**6 herramientas bioinformaticas** ejecutadas como workflow agentico:

1. **Consulta DB** — 4,600+ AMPs de modlAMP + 27 curados antifungicos
2. **Generador** — variantes optimizadas (K/R/L/W, carga, truncamiento)
3. **ESM-2 Embeddings** — protein language model, 650M params, 1280-dim
4. **Propiedades** — 12 descriptores bioquimicos (carga, GRAVY, pI, MW...)
5. **Clasificador ML** — Random Forest, AUC 0.977, 5-fold CV estratificado
6. **Toxicidad** — screening rule-based (GRAVY, carga, largo, Cys, WxxW)

**Score:** `0.35*AMP + 0.25*carga + 0.20*anfipacidad + 0.10*estabilidad + 0.10*(1-tox)`
""",

        # ── Welcome Screen ──
        "hero_title": "De tu cultivo a la solucion biologica",
        "hero_description": """
AgriAMP es un agente de bioinformatica que disena <b>peptidos antimicrobianos</b>
(biopesticidas naturales) contra patogenos de cultivos usando inteligencia artificial.
Describi tu problema — el agente ejecuta un pipeline completo de 6 pasos y
entrega candidatos peptidicos rankeados para sintesis.
""",
        "hero_esm2": "parametros ESM-2",
        "hero_auc": "AUC clasificador",
        "hero_seqs": "secuencias entrenamiento",
        "hero_tools": "tools bioinformaticos",
        "problem": "Problema",
        "problem_desc": """
Argentina usa **500M+ litros/kg de agroquimicos/ano**. Los patogenos generan resistencia.
La UE endurece bans. Los agronomos no tienen acceso a bioinformatica.
""",
        "solution": "Solucion",
        "solution_desc": """
**Peptidos antimicrobianos** (AMPs) — armas del sistema inmune innato con
3+ mil millones de anos de evolucion. Atacan la membrana, no una via metabolica.
""",
        "agentic_pipeline": "Pipeline Agentico",
        "pipeline_desc": """
**6 herramientas** orquestadas como workflow de IA:
consulta DB → embeddings ESM-2 → propiedades → clasificador ML → generador → toxicidad.
""",
        "select_pathogen_info": "Selecciona un patogeno en la barra lateral y presiona **Ejecutar Pipeline AgriAMP** para comenzar.",

        # ── Tab Names ──
        "tab_real_case": "Caso Real",
        "tab_top_candidates": "Top Candidatos",
        "tab_property_analysis": "Analisis de Propiedades",
        "tab_sequence_viewer": "Visor de Secuencias",
        "tab_benchmark": "Benchmark vs SOTA",
        "tab_ml_validation": "Validacion ML",
        "tab_export": "Exportar",

        # ── Pipeline Execution ──
        "agentic_workflow": "Workflow Agentico",
        "duration": "Duracion",
        "running_pipeline": "Ejecutando pipeline AgriAMP...",
        "pipeline_completed": "Pipeline completado en {dur:.1f}s — {n} candidatos analizados",
        "pipeline_error": "El pipeline encontro errores. Revisa el log de arriba.",
        "precomputed_loaded": "Resultados pre-computados cargados.",
        "workflow_steps_completed": "Workflow Agentico — 6 pasos completados",
        "pipeline_summary": "Pipeline completado en {dur:.1f}s — {n_cands} candidatos analizados, {n_passed} pasaron screening de toxicidad",
        "results_for": "Resultados para *{pathogen}* ({common})",

        # ── Tab: Top Candidates ──
        "no_candidates": "No hay candidatos para mostrar.",
        "candidates_analyzed": "Candidatos analizados",
        "passed_toxicity": "Pasaron toxicidad",
        "avg_charge": "Carga promedio",
        "peptide": "Peptido",
        "tox_risk": "Riesgo Tox.",
        "top10_title": "Top 10 Candidatos por AgriAMP Score",
        "threshold_line": "Umbral candidato prometedor (0.70)",
        "candidate_table": "Tabla de Candidatos",

        # ── Tab: Property Analysis ──
        "no_data": "No hay datos para analizar.",
        "radar_title": "Radar: Top Candidato vs Perfil AMP Ideal",
        "charge": "Carga",
        "amphipathicity": "Anfipacidad",
        "stability": "Estabilidad",
        "selectivity": "Selectividad",
        "low_toxicity": "Baja toxicidad",
        "ideal_amp_profile": "Perfil AMP Ideal",
        "feature_importance": "Feature Importance del Clasificador",
        "top10_features": "Top 10 Features Predictivas",
        "no_feature_importance": "Feature importances no disponibles (scoring por propiedades).",
        "property_distributions": "Distribuciones de Propiedades",
        "net_charge_title": "Carga Neta (pH 7)",
        "min_ideal": "Min ideal (+2)",
        "max_ideal": "Max ideal (+9)",
        "hydrophobic_moment_title": "Momento Hidrofobico",
        "amphipathicity_threshold": "Umbral anfipacidad",
        "molecular_weight_title": "Peso Molecular (Da)",

        # ── Tab: Sequence Viewer ──
        "no_sequences": "No hay secuencias para visualizar.",
        "select_peptide": "Seleccionar peptido",
        "colored_sequence": "Secuencia coloreada",
        "hydrophobic": "Hidrofobico",
        "cationic": "Cationico (+)",
        "anionic": "Anionico (-)",
        "polar": "Polar",
        "properties": "Propiedades",
        "net_charge": "Carga neta",
        "molecular_weight": "Peso molecular",
        "hydrophobic_moment": "Momento hidrofobico",
        "isoelectric_point": "Punto isoelectrico",
        "boman_index": "Indice Boman",
        "toxicity": "Toxicidad",
        "score_composition": "Composicion del AgriAMP Score",
        "amp_prob_component": "AMP Probability (35%)",
        "charge_component": "Carga neta (25%)",
        "amphipathicity_component": "Anfipacidad (20%)",
        "stability_component": "Estabilidad (10%)",
        "low_tox_component": "Baja toxicidad (10%)",
        "component": "Componente",
        "contribution": "Contribucion",
        "helical_wheel": "Proyeccion Helical Wheel",
        "helical_caption": "Muestra la distribucion anfipatica — residuos hidrofobicos (amarillo) vs cationicos (azul) en una helice alfa",
        "approved": "(aprobado)",
        "flagged": "(flaggeado)",

        # ── Tab: Benchmark ──
        "benchmark_title": "AgriAMP vs Estado del Arte (2024-2026)",
        "benchmark_description": """
Comparacion de metricas contra herramientas publicadas de prediccion de AMPs.
Todas las metricas de AgriAMP son **out-of-fold** (validacion cruzada estratificada 5-fold,
sin data leakage).
""",
        "tool": "Herramienta",
        "year": "Ano",
        "method": "Metodo",
        "dataset": "Dataset",
        "sensitivity": "Sensibilidad",
        "specificity": "Especificidad",
        "ref": "Ref",
        "agriamp_ours": "AgriAMP (nuestro)",
        "this_work": "Este trabajo",
        "benchmark_note": """
*Nota: Las metricas NO son directamente comparables entre herramientas (cada una usa datasets diferentes).
amPEPpy reporta AUC 0.99 en su propio dataset custom. AgriAMP usa el benchmark estandar modlAMP.
La comparacion muestra que nuestro pipeline es competitivo en el rango SOTA.*
""",
        "auc_chart_title": "AUC-ROC: AgriAMP vs Herramientas Publicadas",
        "differentiator_title": "Diferenciador de AgriAMP",
        "differentiator_content": """
Las herramientas comparadas son **clasificadores**: reciben una secuencia y predicen
si es AMP. AgriAMP es un **pipeline agentico completo** que:

1. Consulta bases de datos de AMPs conocidos para el patogeno especifico
2. Genera variantes optimizadas con mutaciones dirigidas
3. Genera embeddings con ESM-2 (650M params)
4. Calcula 12 propiedades bioquimicas
5. Clasifica con Random Forest (AUC competitivo)
6. Filtra por toxicidad y selectividad

**Ninguna otra herramienta ofrece este workflow integrado y accesible para agronomos.**
""",
        "full_metrics": "Metricas Completas de AgriAMP",
        "training_data": "Datos entrenamiento",
        "cv_folds": "CV Folds",
        "stratified": "5 (estratificado)",

        # ── Tab: ML Validation ──
        "no_validation": "No hay metricas de validacion.",
        "property_scoring_info": "Se uso scoring basado en propiedades (no ML). Metricas de validacion cruzada no disponibles.",
        "classifier_metrics": "Metricas del Clasificador (Out-of-Fold)",
        "data_caption": "Datos: {pos} AMPs + {neg} non-AMPs | 5-fold CV estratificado | AUC std: \u00b1{std:.3f}",
        "confusion_matrix": "Matriz de Confusion (Out-of-Fold)",
        "prediction": "Prediccion",
        "actual": "Real",
        "auc_per_fold": "AUC-ROC por Fold de Cross-Validation",
        "roc_curve": "Curva ROC",

        # ── Tab: Export ──
        "no_export": "No hay resultados para exportar.",
        "export_results": "Exportar Resultados",
        "download_csv": "Descargar CSV completo",
        "download_fasta": "Descargar FASTA (top candidatos)",
        "analysis_summary": "Resumen del Analisis",

        # ── Tab: Real Case ──
        "caso_real_hero": """
<div class="hero-box">
    <h3>Caso Real: Botrytis cinerea en Mendoza</h3>
    <p style="font-size:1.05rem;">
        La podredumbre gris (<i>Botrytis cinerea</i>) es el hongo mas agresivo y
        prevalente en los vinedos de Mendoza. La literatura reporta perdidas de
        hasta <b>50-80% de la cosecha</b> en anos de infeccion severa, con impacto
        significativo en rendimiento y calidad del fruto (PMC, Springer).
    </p>
</div>
""",
        "loss_botrytis": "Perdida por Botrytis",
        "loss_botrytis_help": "Rango en anos severos (PMC, Springer)",
        "wine_exports": "Exportaciones vino ARG",
        "wine_exports_help": "Mendoza = 75% produccion nacional",
        "global_losses": "Perdidas globales/ano",
        "global_losses_help": "Impacto economico global de B. cinerea",
        "agrochemicals": "Agroquimicos ARG/ano",
        "agrochemicals_help": "SPRINT-H2020, Mongabay 2020",
        "weight_loss": "Perdida en peso",
        "weight_loss_help": "Peso de cosecha perdido",
        "banned_pesticides": "Pesticidas ARG banned en UE",
        "banned_pesticides_help": "Acuerdo EU-Mercosur, enero 2026",
        "regulatory_crisis": "Crisis Regulatoria",
        "regulatory_content": """
- **Nuevos limites MRL de importacion UE** (marzo 2026): residuos de clothianidin y thiamethoxam prohibidos en productos importados (ban de uso outdoor desde 2018)
- **Acuerdo EU-Mercosur**: Argentina tiene **106 principios activos prohibidos** en la UE — barrera para exportacion de vinos
- **Resistencia documentada**: mutaciones duales SDHI + QoI en poblaciones de *B. cinerea* a nivel global
- **Biopesticidas basados en peptidos aprobados para vid**: **ninguno** — oportunidad de mercado clara
""",
        "validated_amps": "Validacion con AMPs Publicados",
        "validated_amps_table": """
| AMP | MIC vs *B. cinerea* | Fuente | Referencia |
|-----|---------------------|--------|------------|
| **Epinecidin-1** | 12.5 umol/L | *Epinephelus coioides* (mero) | Food Chemistry, 2022 |
| **EPI-4** (variante optimizada) | **6 umol/L** (2x mejor) | Sintetico | ACS JAFC, 2025 |
| **Rs-AFP2** | 3 umol/L | *Raphanus sativus* (rabano) | Plant defensin literature |
| **NaD1** | 2 umol/L | *Nicotiana alata* (tabaco) | Plant defensin literature |

EPI-4 es una variante optimizada de Epinecidin-1 con sustituciones en la superficie polar
(Pan et al., JAFC 2025). Nuestro generador de variantes aplica estrategias similares:
sustitucion con residuos cationicos (+K/R) para aumentar carga y actividad antimicrobiana.
""",
        "epinecidin_in_results": "Epinecidin en nuestros resultados",
        "epinecidin_row": "- **{name}**: AgriAMP Score **{score:.4f}** | Carga {charge:+.1f} | AMP prob {prob:.3f} | Toxicidad {tox:.2f} {status}",
        "next_step": "Proximo Paso: Validacion Experimental",
        "next_step_content": """
Los candidatos de AgriAMP son **candidatos in silico para validacion experimental**.
El siguiente paso es **sintesis quimica** ($80-250/peptido screening) y ensayos de
inhibicion in vitro contra *B. cinerea*. Socio natural: **INTA Mendoza** (Laboratorio
de Fitopatologia, equipo de Dra. Georgina Escoriaza).
""",
    },
}
