# PredictIQ - Day 70 Demo and Verification

## Product

PredictIQ is an educational historical bicycle
rental-demand exploration application.

The backend uses FastAPI and the previously
trained Day 66 Random Forest pipeline.

The responsive frontend uses HTML, CSS and JavaScript.

## Verified Results

Automated tests: 57 passed.

Real API hourly predictions: 24 successful.

Historical example results:

- Lowest estimated hourly rentals: 2.3
- Highest estimated hourly rentals: 226.0
- Peak estimated hour: 08:00

These estimates are specific to the example
historical scenario, not live demand measurements.

## Start the Application

Run from the repository root:

    python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

Website:

    http://127.0.0.1:8000/

Interactive API documentation:

    http://127.0.0.1:8000/docs

## Two-Minute Demonstration

00:00 Introduce PredictIQ and its historical dataset.

00:20 Explain the Random Forest and validation MAE.

00:40 Select a scenario and click Predict.

01:00 Explain how FastAPI loads the saved model.

01:20 Change an input and obtain another prediction.

01:35 Generate the 24-hour demand profile.

01:50 Explain the limitations and intended use.

## Failure Cases Verified

1. Invalid input:
   HTTP 422 validation response.

2. Unavailable prediction model:
   HTTP 503 response.

3. Missing frontend:
   HTTP 404 response.

4. Missing model during startup:
   Application startup fails clearly.

Tests simulate failures without deleting
the actual model or frontend.

## Model Information

Historical validation MAE: approximately 47.68.

Historical validation RMSE: approximately 74.14.

Dataset: UCI Bike Sharing, Washington DC, 2011-2012.

The held-out test period remains unused.

## Important Deployment Limitation

This application currently runs locally.

The GitHub repository includes source code
but excludes the raw dataset and trained
joblib model.

A new installation must prepare the dataset
and regenerate the trusted Day 66 model.

Never load untrusted joblib or pickle artifacts.

GitHub publication alone does not create
a publicly accessible website.

## Automated Tests

    python -m pytest tests -q

## Related Documentation

- docs/day70_model_card.md
- docs/day66_model_improvement.md
