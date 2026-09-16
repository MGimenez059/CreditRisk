# Notebooks

`01_data_exploration.ipynb` contains an executed historical Phase 2 EDA with saved
outputs. It reads validated Parquet produced by `scripts/ingest_data.py`. Install
the `viz` extra for Matplotlib and a notebook kernel.

Its outputs precede the grouped Phase 3 protocol. They provide historical descriptive
evidence, not current model validation. Re-execution requires the local dataset.
Model comparison and SHAP results are published in the
[evaluation report](../docs/evaluation_report.md) and
[explanation report](../docs/explainability.md); their workflows run through scripts.

Reusable transformations and training belong in `src/credit_risk/ml/`. Application
code never imports notebooks. See [ROADMAP.md](../ROADMAP.md).
