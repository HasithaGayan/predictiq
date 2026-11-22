# PredictIQ — Urban Mobility Demand Intelligence

Project 04 of the HGBP 87-Day AI + Full-Stack Roadmap.

## Goal

Build a professional machine-learning application
that predicts hourly bike rental demand from weather
and calendar features.

## Project Design

Background: #F7FAF9
Primary teal: #0F766E
Text: #12312D

## ML Task

Type: Supervised regression

Target: cnt (total hourly bike rentals)

Important target leakage:
casual + registered = cnt

These count components must NOT be model features.

## Dataset

UCI Machine Learning Repository — Bike Sharing Dataset

Source:
https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset

Citation:
Fanaee-T, H. (2013). Bike Sharing [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5W894

License: CC BY 4.0

Data covers historical 2011–2012 rentals.
The demo must not claim current forecasting accuracy.

## Planned Architecture

Data -> Cleaning -> EDA -> Model Pipeline
     -> Model Evaluation -> FastAPI -> Frontend

## Development Status

Day 63: Initial project setup and data preparation.

A trained model, production API, deployment and
evaluation metrics have NOT yet been completed.

## Day 64 - EDA and Baseline

A leakage-safe chronological split, training-only
exploratory analysis, four visualizations, and a
scikit-learn mean baseline are implemented.

See [Day 64 results](docs/day64_eda_baseline.md).

This is a baseline, not a production-ready ML model.

## Day 65 - Supervised Models

Trained and evaluated a mean baseline, Ridge regression
and Random Forest using a leakage-safe chronological split.

See [Day 65 model comparison](docs/day65_model_training.md).

The saved pipeline remains local. The held-out test
period has not been used for model selection.

## Day 66 - Model Improvement

Implemented training-only chronological cross-validation,
two engineered features, and controlled Random Forest
hyperparameter comparisons.

See [Day 66 model improvement](docs/day66_model_improvement.md).

The selected pipeline is stored locally.
The held-out test set remains unused.

## Day 67 - Unsupervised Learning

Implemented training-only K-Means clustering,
feature scaling, PCA visualization, cluster inspection,
and automated correctness tests.

See [Day 67 clustering and PCA](docs/day67_unsupervised.md).

The clustering experiment is exploratory, not a
production demand-prediction model.

## Day 68 - Neural Networks

Implemented a PyTorch neural-network training
experiment with two learning rates, a forward pass,
backpropagation, loss tracking and validation evaluation.

See [Day 68 neural-network results](docs/day68_neural_network.md).

The existing Random Forest remains the leading
supervised reference model.
