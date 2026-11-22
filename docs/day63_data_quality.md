# PredictIQ — Day 63 Data Quality Report

## Project

PredictIQ — Urban Mobility Demand Intelligence

## Dataset

UCI Bike Sharing Dataset (historical 2011–2012 data).

Prediction target:
cnt — total hourly bike rental demand.

## Dataset Verification

- Original records: 17,379
- Original columns: 17
- Missing cells: 0
- Duplicate rows: 0
- Processed records: 17,379

## Reproducible Cleaning

The pipeline:

1. Reads the original hourly CSV.
2. Checks required columns.
3. Rejects missing values.
4. Validates dates and categories.
5. Rejects duplicate date/hour combinations.
6. Validates rental count consistency.
7. Creates time-based features.
8. Excludes identifier and leakage columns.
9. Saves cleaned data separately.
10. Generates a JSON quality report.

## Feature Engineering

is_weekend:
Indicates weekend observations.

hour_sin and hour_cos:
Represent the circular nature of time.

For example, 23:00 and 00:00 are close
to each other despite their numeric difference.

## Preventing Data Leakage

Target: cnt

Excluded:
- casual
- registered
- instant

Since casual + registered = cnt,
including these columns would directly
reveal the prediction target.

## Safety

The raw dataset is never overwritten.

The training, validation and test data split
has not yet been performed.

No ML model has been trained yet.

## Automated Testing

The cleaning pipeline is covered by
eight pytest tests, including invalid values,
duplicate records and raw-data protection.

## Next Milestone

Day 64 — Exploratory Data Analysis,
visualizations, baseline preparation and
leakage-safe training/validation/test planning.

## Dataset Credit

Fanaee-T, H. (2013).
Bike Sharing [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5W894

License: CC BY 4.0.
