# PredictIQ — Day 70 Model Card

## Model

Selected Day 66 Random Forest regression pipeline.

## Purpose

Estimate historical hourly bike-rental demand
under user-specified weather and calendar conditions.

This is an educational interactive demonstration.

## Dataset

UCI Bike Sharing Dataset.

Location: Washington, DC, USA.

Historical period: 2011–2012.

Training rows: 12,139.
Validation rows: 2,640.
Held-out test rows: 2,600.

The held-out test set has not been evaluated.

## Input Features

The trained pipeline receives 15 features.

The public API accepts 11 input fields and calculates
workingday, is_weekend, hour_sin and hour_cos.

Normalized weather inputs follow the training
dataset's representation.

## Validation Performance

Validation MAE: approximately 47.68 rentals.
Validation RMSE: approximately 74.14 rentals.

These are historical validation results,
not guaranteed error bounds for future predictions.

The validation data was used during model selection.
The independent test period remains unused.

## Appropriate Use

Historical educational scenario exploration.
Model comparisons.
Learning about supervised machine learning.
Demonstrating API-based model inference.

## Inappropriate Use

Live operational bike allocation.
Current city-demand forecasting.
Financial or safety-critical decisions.
Claims of real-time accuracy.

## Data and Model Limitations

- Historical data from only 2011–2012.
- Washington, DC-specific patterns.
- No live weather or rental information.
- Hypothetical scenarios may combine calendar
  fields that did not occur together in reality.
- No confidence or prediction interval.
- Some user inputs may differ from conditions
  represented in the training dataset.
- No independent held-out test evaluation yet.
- Not a publicly deployed service.

## Local Model Artifact

reports/generated/day66/selected_pipeline.joblib

The trained artifact stays in the ignored local
generated directory rather than GitHub.

It must be regenerated or provided separately
in a trusted deployment environment.

Never load untrusted pickle or joblib artifacts.

## API

GET /api/health

GET /api/model-info

POST /api/predict

## Failure Cases Tested

1. Invalid hour or input:
   The API rejects invalid values with HTTP 422.

2. Model temporarily unavailable:
   The API returns HTTP 503.

3. Frontend missing:
   The server returns HTTP 404.

4. Saved model missing during startup:
   Application startup fails instead of
   producing fabricated predictions.

## Reproducibility

Use the documented Python environment,
install project dependencies, prepare the
original dataset and regenerate the Day 66
model artifact before starting the API.

Run the automated test suite:

    python -m pytest tests -q

Start the local app:

    python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

Open:

    http://127.0.0.1:8000/

## Dataset Attribution

Fanaee-T, H. (2013).
Bike Sharing Dataset.
UCI Machine Learning Repository.

https://doi.org/10.24432/C5W894
