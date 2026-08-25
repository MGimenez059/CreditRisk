# notebooks/

`01_data_exploration.ipynb` (Phase 2) has cells but no run outputs — it
needs the real dataset at `data/raw/credit_risk_dataset.csv` (validated via
`scripts/ingest_data.py` first) to actually produce anything. Phase 4 adds
a model comparison notebook; Phase 5 adds `03_model_analysis.ipynb` (SHAP
summary plots, referenced from the README's Explainability section).

Notebooks here are for exploration only — no notebook is ever imported by
`src/credit_risk`, per CODESTYLE.md's layering rules.
