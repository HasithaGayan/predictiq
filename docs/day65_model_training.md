# PredictIQ — Day 65 Supervised Machine Learning

## Objective

Train and compare supervised regression models
for hourly bike rental demand.

## Dataset

UCI Bike Sharing Dataset, 2011–2012.

Target: cnt

Excluded features:
- casual
- registered
- instant

## Data Split

Training records: 12139
Validation records: 2640
Held-out test records: 2600

Training data ends: 2012-05-25
Validation data ends: 2012-09-12

All candidates use the same chronological split.

## Models Evaluated

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| mean_baseline | 194.5179 | 254.6071 |
| ridge | 111.2566 | 142.3599 |
| random_forest | 49.8529 | 77.2611 |

## Best Current Model

Selected model: random_forest

Selection metric: validation_mae

Validation MAE: 49.8529

Validation RMSE: 77.2611

## Largest Validation Errors

- 2012-05-28 at 8:00 | Actual: 155 | Predicted: 594.97 | Error: 439.97
- 2012-07-04 at 8:00 | Actual: 153 | Predicted: 585.04 | Error: 432.04
- 2012-09-03 at 8:00 | Actual: 127 | Predicted: 557.72 | Error: 430.72
- 2012-08-01 at 18:00 | Actual: 891 | Predicted: 475.69 | Error: 415.31
- 2012-08-19 at 14:00 | Actual: 149 | Predicted: 508.57 | Error: 359.57

These errors require investigation.
Their causes have not yet been established.

## Engineering Decisions

- Preprocessing is fitted only on training data.
- Pipelines prevent inconsistent transformations.
- Models are compared on the same validation period.
- Random Forest uses a fixed random seed.
- The best pipeline is saved locally with joblib.
- Model versions and source-data hash are recorded.
- Test-set labels are not used for model selection.

## Reproducibility

Cleaned dataset SHA256:
6f91fa426fdc195372c1388a68b2883f4b70705dd0573f3c1ad8ada17276370f

scikit-learn:
1.9.1

pandas:
3.1.1

NumPy:
2.5.5

Run:

python -m src.predictiq.day65_train

The generated model binary is excluded from Git.
Regenerate it using the source code and dependencies.

Only load joblib files from trusted sources.

## Limitations

- Historical 2011–2012 data.
- Validation results are not final test performance.
- Some hours have large prediction errors.
- Modern demand accuracy is not established.
- The ML API and public frontend remain unfinished.

## Next

Day 66: Time-series cross-validation, controlled
feature experiments and hyperparameter tuning.

Keep the held-out test period untouched.

## Dataset Credit

Fanaee-T, H. (2013).
Bike Sharing [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5W894
