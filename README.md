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
