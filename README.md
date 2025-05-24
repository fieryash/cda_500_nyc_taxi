# NYC Taxi Demand Prediction Pipeline 🚖

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Hopsworks](https://img.shields.io/badge/Hopsworks-Feature%20Store-brightgreen)](https://www.hopsworks.ai/)

## Overview

This repository provides a comprehensive pipeline for **predicting New York City taxi demand** using time-series and machine learning techniques. It automates data ingestion, feature engineering, model training, and deployment, leveraging the Hopsworks Feature Store for robust data management.

## Features

- **Automated Data Fetching:** Download raw NYC taxi ride data by month and year, storing in structured directories.
- **Feature Engineering Pipeline:** Transform raw data into time-series features using Jupyter Notebooks and Python scripts.
- **Model Training and Prediction:** Implements various models (ARMA, ARIMA, Prophet, LGBMRegressor) for predicting hourly taxi demand.
- **MLOps Integration:** Uses Hopsworks for feature storage, retrieval, and model artifact management.
- **End-to-End Notebooks:** Modular notebooks for each pipeline stage (fetch, engineer, train, predict).
- **Extensible and Reproducible:** Isolated Python environments and clear setup steps.

## Project Structure

```
├── notebooks/
│   ├── 01_fetch_data.ipynb           # Fetch raw taxi ride data
│   ├── 10_lgm_with_fe.ipynb          # LightGBM with feature engineering
│   ├── 12_load_features_hopsworks.ipynb # Feature loading to Hopsworks
│   ├── 13_feature_pipeline.ipynb     # Full feature engineering pipeline
│   ├── 14_model_training_pipeline.ipynb # Model training workflow
│   ├── 15_predict_using_hopsworks_model.ipynb # Model inference using Hopsworks
│   ├── 17_fetch_predictions.ipynb    # Fetch predictions from registry
│   └── 22_ARMA_ARIMA_Prophet.ipynb   # Classical time-series models
├── src/
│   ├── feature_pipeline.py           # Python pipeline for data/feature flow
│   ├── inference.py                  # Model loading, prediction, and utilities
│   └── pipeline_utils.py             # Helper functions for feature engineering
├── requirements_with_version.txt     # Python dependencies (with versions)
├── todo.md                           # Project setup and development TODOs
```

## Quick Start

### 1. Environment Setup

- **Install Python 3.10+** (Anaconda recommended)
- **Create a virtual environment** (e.g., with `poetry` or `venv`)
- **Install dependencies:**
  ```bash
  pip install -r requirements_with_version.txt
  ```

### 2. Directory Structure

Create the following directories for data management:
```
/data
  /raw        # For downloaded raw parquet files
  /processed  # For processed features
/models       # For trained model artifacts
```

### 3. Fetch Data

Use the notebook `notebooks/01_fetch_data.ipynb` to download monthly taxi ride data:
```python
fetch_raw_data(2023, 1)
# Data saved to ../data/raw/rides_2023_01.parquet
```

### 4. Feature Engineering

Run the pipeline (`src/feature_pipeline.py` or `notebooks/13_feature_pipeline.ipynb`) to process and transform raw data into features suitable for machine learning.

### 5. Model Training & Prediction

- Train models using provided notebooks/scripts.
- Save models to `/models`, or directly to Hopsworks Model Registry.
- Fetch and visualize predictions via `notebooks/17_fetch_predictions.ipynb`.

## Hopsworks Integration

- **Login and Connect:**
  ```python
  import hopsworks
  project = hopsworks.login(project='cda_500_nyc_taxi', api_key_value='YOUR_API_KEY')
  feature_store = project.get_feature_store()
  ```
- **Feature Store:** Used for storing and retrieving processed features for training and inference.
- **Model Registry:** Save model artifacts for later deployment.

## Example: Model Prediction

```python
from src.inference import load_model_from_registry, get_model_predictions

model = load_model_from_registry()
features = ... # Load features DataFrame
predictions = get_model_predictions(model, features)
```

## Development Notes

- Check `todo.md` for setup steps and development progress.
- Modular organization: Each notebook/script covers a distinct pipeline step.
- Extensive use of logging for transparency and debugging.

## Contributing

Contributions and issues are welcome! Please open an issue or pull request if you have suggestions, improvements, or bug reports.

---

**Author:** [@fieryash](https://github.com/fieryash)

**License:** MIT

---

> _“Making NYC transportation smarter, one ride at a time!”_ 🚕
