# Model Card – E-commerce Purchase Behavior Prediction

- **Experiment ID:** `20261121_001601_870021`
- **Trained:** 2026-11-21T00:16:01
- **Task:** classification – predict `will_purchase` (`Yes` / `No`)
- **Best Model:** Logistic Regression
- **Best Parameters:** {'C': 0.1}
- **Data:** 24,000 training rows, 6,000 test rows
- **Selection Metric:** f1

## Test-Set Performance

| Metric | Value |
|---|---|
| accuracy | 0.8645 |
| precision | 0.3108 |
| recall | 0.6462 |
| f1 | 0.4197 |
| roc_auc | 0.8133 |
| pr_auc | 0.3964 |

## Model Comparison (Cross-Validation)

| Model | f1 (mean ± std) | ROC-AUC (mean ± std) |
|---|---|---|
| Logistic Regression | 0.4409 ± 0.0086 | 0.835 ± 0.0081 |
| Random Forest | 0.4332 ± 0.0127 | 0.8319 ± 0.0094 |
| Gradient Boosting | 0.3499 ± 0.0268 | 0.8381 ± 0.0082 |
| Decision Tree | 0.3006 ± 0.0219 | 0.62 ± 0.0129 |

## Methodological Notes & Limitations

- **Leakage Prevention:** Features are extracted from the first 3 non-transaction events of each session plus strictly earlier sessions. Sessions that already purchased at or before event 3 are excluded.
- **Imbalanced Target:** Only ~7.6% of eligible sessions convert after the snapshot; evaluate with F1, Recall, Precision, PR-AUC, and ROC-AUC rather than accuracy alone.
