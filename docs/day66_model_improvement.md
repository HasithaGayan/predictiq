# PredictIQ — Day 66 Model Improvement

## Objective

Improve hourly bike rental demand predictions using
chronological cross-validation, feature engineering,
and controlled Random Forest experiments.

## Dataset and Leakage Protection

Dataset: UCI Bike Sharing Dataset (2011-2012).
Target: cnt — total hourly bike rental demand.

Excluded target-leakage features:
- casual
- registered

The identifier instant is also excluded.

Cross-validation uses training dates only.
Preprocessing is fitted within each training fold.
The final 2,600 test rows remain held out.

## New Features

commuter_peak:
Identifies common commuting hours on working days.

temperature_humidity:
Multiplies temperature and humidity to represent
a possible interaction between weather conditions.

These features are created from model inputs, not
from the target value.

## Training-Only Cross-Validation

Three chronological expanding folds were used.

| Configuration | Engineered | Trees | Depth | Leaf | CV MAE | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|
| standard | False | 40 | 12 | 3 | 54.981 | 3.88 |
| features_only | True | 40 | 12 | 3 | 54.8616 | 4.05 |
| deeper | True | 70 | 18 | 2 | 54.4441 | 13.66 |
| regularized | True | 70 | 12 | 5 | 56.1509 | 11.61 |

Best cross-validation configuration:
deeper

## Validation Comparison

| Model | MAE | RMSE | Runtime (s) |
|---|---:|---:|---:|
| day65_reference | 49.8529 | 77.2611 | 21.03 |
| day66_cv_selected | 47.6834 | 74.1438 | 21.38 |

Mean baseline MAE:
194.5179

Selected model:
day66_cv_selected

Selected validation MAE:
47.6834

Selected validation RMSE:
74.1438

Lower MAE and RMSE indicate lower prediction
error on this historical validation period.

## Five Largest Validation Errors

- 2012-08-01 at 18:00 — Actual: 891, Predicted: 409.92, Absolute error: 481.08
- 2012-07-10 at 18:00 — Actual: 819, Predicted: 426.16, Absolute error: 392.84
- 2012-08-14 at 17:00 — Actual: 812, Predicted: 421.38, Absolute error: 390.62
- 2012-07-16 at 17:00 — Actual: 755, Predicted: 405.54, Absolute error: 349.46
- 2012-07-07 at 17:00 — Actual: 279, Predicted: 624.22, Absolute error: 345.22

These errors show that the model still struggles
on some individual hours.

Their exact causes require further investigation.

## Engineering Decisions

- Chronological cross-validation avoids future folds
  being used to train earlier predictions.
- Preprocessing is fitted only on training folds.
- New features are computed within the pipeline.
- Configurations are compared using CV MAE.
- The Day 65 reference model is preserved.
- Final comparison uses the validation period.
- The held-out test period is not evaluated here.
- The selected pipeline is saved locally.

## Reproducibility

Run from the PredictIQ repository root:

python -m src.predictiq.day66_experiment

Run tests:

python -m pytest tests -v

Saved model:
reports/generated/day66/selected_pipeline.joblib

Experiment report:
reports/generated/day66/experiment_report.json

Generated model binaries are excluded from Git.
Only load trusted joblib model files.

## Limitations

- Historical data covers 2011-2012.
- Validation performance is not final test accuracy.
- Some peak-demand predictions have large errors.
- Modern real-world forecasting remains unverified.
- The FastAPI application and web interface are
  not yet completed.

## Next Roadmap Milestones

Day 67: Unsupervised learning and PCA.

Day 68: Neural-network fundamentals with PyTorch.

Day 69: Embeddings and transformer concepts.

Day 70: ML/FastAPI proof application checkpoint.

## Dataset Attribution

Fanaee-T, H. (2013).
Bike Sharing Dataset.
UCI Machine Learning Repository.
https://doi.org/10.24432/C5W894
