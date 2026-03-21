# AgriAMP

**Pipeline agentico de IA para el diseno de peptidos antimicrobianos (AMPs) contra patogenos agricolas.**

[Demo en vivo](https://agriamp.streamlit.app/)

---

## El problema

- **500M+ litros/kg** de agroquimicos por ano en Argentina (SPRINT-H2020)
- **Botrytis cinerea** causa perdidas de hasta 71% en racimos de uva en Mendoza (CONICET)
- **106 principios activos** usados en Argentina estan prohibidos en la UE
- Los patogenos desarrollan **resistencia** a pesticidas quimicos que atacan una sola via metabolica
- **No existen fungicidas basados en peptidos** aprobados para vid — hay un gap de mercado claro

## La solucion

AgriAMP es un pipeline end-to-end que toma un patogeno agricola como input y genera candidatos de peptidos antimicrobianos optimizados para validacion experimental. El usuario describe su problema (ej. "Botrytis en mi vinedo") y el sistema ejecuta 6 herramientas autonomas.

## Pipeline de 6 pasos

```
Patogeno → [1] Consulta BD → [2] Embeddings ESM-2 → [3] Clasificador RF
                                                            ↓
         Candidatos ← [6] Scoring ← [5] Toxicidad ← [4] Generador variantes
```

1. **Consulta de bases de datos** — AMPs curados con actividad contra el patogeno target
2. **Embeddings ESM-2** — Representaciones de 1280 dimensiones via protein language model (650M params, 65M secuencias UniRef50)
3. **Clasificador Random Forest** — 200 arboles, 89 features (PCA embeddings + propiedades bioquimicas + composicion AA)
4. **Generador de variantes** — Mutaciones puntuales, optimizacion de carga, truncamientos N/C-terminal
5. **Screening de toxicidad** — Filtros rule-based (GRAVY, carga, largo, cisteina, motivos hemoliticos)
6. **Scoring compuesto** — `0.35*AMP_prob + 0.25*carga + 0.20*anfipacidad + 0.10*estabilidad + 0.10*(1-toxicidad)`

## Metricas (5-fold CV estratificado, 5200 secuencias)

| Metrica | Valor |
|---------|-------|
| AUC-ROC | 0.977 |
| MCC | 0.858 |
| F1-Score | 0.929 |
| Sensibilidad | 0.932 |
| Especificidad | 0.927 |
| Accuracy | 92.9% |

Entrenado sobre 2,600 AMPs + 2,600 non-AMPs del dataset modlAMP con embeddings pre-cacheados de ESM-2.

## Validacion con datos reales

- **Epinecidin-1** (MIC 12.5 umol/L vs B. cinerea, publicado en J. Agric. Food Chem.) aparece en los resultados
- **EPI-4** (variante con +K terminal, MIC 6.0 umol/L) obtiene score superior (0.756 vs 0.712)
- El pipeline identifica correctamente que la variante optimizada es mejor que el original

## Stack tecnico

- **ESM-2** (Meta, `facebook/esm2_t33_650M_UR50D`) — protein language model
- **scikit-learn** — Random Forest, StratifiedKFold, metricas out-of-fold
- **Streamlit** — interfaz web interactiva
- **Plotly / Matplotlib** — visualizaciones (radar, ROC, confusion matrix, helical wheel)
- **PyTorch + HuggingFace Transformers** — inference ESM-2
- **modlAMP** — dataset de entrenamiento AMP vs non-AMP

## Uso local

```bash
# Clonar
git clone https://github.com/tu-usuario/agriamp.git
cd agriamp

# Instalar dependencias
pip install -r requirements.txt

# Pre-computar embeddings (una sola vez, requiere GPU)
python precompute_embeddings.py

# Ejecutar pipeline completo
python agent.py

# Lanzar interfaz web
streamlit run app.py
```

**Requisitos:** Python 3.10+, GPU con CUDA (para embeddings ESM-2). Sin GPU, la app usa datos pre-computados.

## Estructura del proyecto

```
AGRIAMP/
├── app.py                    # Interfaz Streamlit (7 tabs)
├── agent.py                  # Orquestador del pipeline agentico
├── precompute_embeddings.py  # Cache de embeddings ESM-2
├── requirements.txt          # Dependencias
├── logo_agriamp.png          # Logo
├── GUION_DEMO.html           # Guion de la demo (3 min)
├── BATTLECARDS.html          # Datos y respuestas para jurados
├── tools/
│   ├── __init__.py           # BaseTool + ToolResult
│   ├── data_query.py         # [1] Consulta BD de AMPs curados
│   ├── embeddings.py         # [2] ESM-2 protein embeddings
│   ├── classifier.py         # [3] Random Forest + metricas
│   ├── generator.py          # [4] Generador de variantes
│   ├── toxicity.py           # [5] Screening de toxicidad
│   └── properties.py         # Propiedades bioquimicas
└── data/
    ├── modlamp_embeddings.npz      # Cache de embeddings (gitignored)
    └── precomputed/
        ├── botrytis_cinerea.json
        ├── fusarium_graminearum.json
        ├── xanthomonas_citri.json
        └── ralstonia_solanacearum.json
```

## Patogenos soportados

| Patogeno | Tipo | Cultivo afectado |
|----------|------|-----------------|
| Botrytis cinerea | Hongo | Vid, frutilla, tomate |
| Fusarium graminearum | Hongo | Trigo, maiz |
| Xanthomonas citri | Bacteria | Citricos |
| Ralstonia solanacearum | Bacteria | Papa, tomate |

---

Desarrollado para el **Aleph Hackathon M26** — Track Biotech, marzo 2026.
