# Notebooks

`01_data_exploration.ipynb` contains an executed historical Phase 2 EDA with saved
outputs. It reads validated Parquet produced by `scripts/ingest_data.py`. Install
the `viz` extra for Matplotlib and a notebook kernel.

Its outputs precede the grouped Phase 3 protocol. They provide historical descriptive
evidence, not current model validation. Re-execution requires the local dataset.
Model comparison and SHAP notebooks may be added when those phases produce results;
there is no requirement to create empty notebooks in advance.

Reusable transformations and training belong in `src/credit_risk/ml/`. Application
code never imports notebooks. See [ROADMAP.md](../ROADMAP.md).
