# Model Card – E-commerce Purchase Behavior Prediction

- **Trained:** 2026-11-20T23:20:22
- **Task:** classification – predict `will_purchase` (`Yes` / `No`)
- **Best Model:** Logistic Regression
- **Best Parameters:** {'C': 3.0}
- **Data:** 24,000 training rows, 6,000 test rows (Retailrocket early-session snapshots)
- **Selection Metric:** f1 (5-fold Stratified CV)

## Test-Set Performance

| Metric | Value |
|---|---|
| accuracy | 0.865 |
| precision | 0.3118 |
| recall | 0.6462 |
| f1 | 0.4206 |
| roc_auc | 0.8165 |
| pr_auc | 0.3758 |

## Model Comparison (Cross-Validation)

| Model | f1 (mean ± std) | ROC-AUC (mean ± std) |
|---|---|---|
| Logistic Regression | 0.4266 ± 0.0087 | 0.8295 ± 0.0057 |
| Random Forest | 0.4227 ± 0.0117 | 0.829 ± 0.0119 |
| Gradient Boosting | 0.3466 ± 0.024 | 0.8399 ± 0.0071 |
| Decision Tree | 0.3092 ± 0.0164 | 0.624 ± 0.0105 |

## Methodological Notes & Limitations

- **Leakage Prevention:** Features are extracted from the first 3 non-transaction events of each session plus strictly earlier sessions. Sessions that already purchased at or before event 3 are excluded.
- **Scope:** Applies to engaged multi-event sessions (≥3 interactions); single-click bounce sessions lack an early observation window.
- **Imbalanced Target:** Only ~7.6% of eligible sessions convert after the snapshot; evaluate with F1, Recall, Precision, PR-AUC, and ROC-AUC rather than accuracy alone.
