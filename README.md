# Project Name: Enviornmental Air Quality Automated  MLOps Pipeline

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Collection & Versioning](#data-collection--versioning)
- [Model Training & Experiment Tracking](#model-training--experiment-tracking)
- [API Deployment & Monitoring](#api-deployment--monitoring)
- [Data Processing & Visualization](#data-processing--visualization)
- [Achievements & Project Outcomes](#achievements--project-outcomes)

## Overview
This project implements a complete MLOps pipeline that covers data ingestion, versioning, model training, deployment, and monitoring. The system automates the data pipeline, ensuring reproducibility, model tracking, and performance monitoring.

## Architecture
```
├── src
│   ├── api
│   │   ├── app.py                   # FastAPI application
│   │   ├── metricstest.py           # Unit tests for API metrics
│   ├── models
│   │   ├── train.py                 # Model training using TensorFlow/Keras
├── data
│   ├── raw                          # Versioned raw data managed with DVC
│   ├── processed                    # Processed data for model training
├── docker
│   ├── Dockerfile                   # Docker configuration for deployment
├── monitoring
│   ├── grafana
│   │   ├── dashboards
│   │   │   ├── environmental_monitoring.json # Grafana dashboard for visualization
├── scripts
│   ├── schedule_collection.py       # Automated data collection script
├── .env                             # Environment variables for configuration
├── requirements.txt                 # Dependencies list
├── README.md                        # Project documentation
```

## Technology Stack
- **Programming Language:** Python
- **Frameworks & Libraries:**
  - FastAPI (REST API)
  - TensorFlow/Keras (Model Training)
  - scikit-learn (Evaluation Metrics)
  - pandas, numpy (Data Processing)
  - matplotlib, seaborn (Visualization)
  - prometheus_client (Metrics & Monitoring)
  - MLflow (Experiment Tracking)
  
- **Infrastructure & DevOps:**
  - Docker (Containerization)
  - Prometheus & Grafana (Monitoring & Visualization)
  - DVC (Data Versioning)
  - Git (Version Control & Automation)

## Setup and Installation
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/your-repo/mlops-project.git
   cd mlops-project
   ```
2. **Set Up a Virtual Environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set Up Environment Variables:**
   ```sh
   cp .env.example .env  # Modify .env with appropriate values
   ```
5. **Run the API:**
   ```sh
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```

## Data Collection & Versioning
- **Automated Data Collection:** A scheduled process in `scripts/schedule_collection.py` collects data periodically.
- **Versioning with DVC:** Raw data stored in `data/raw/` is version-controlled using DVC.
- **Git Integration:** DVC commits are automated, ensuring data consistency.
  ```sh
  dvc init
  dvc add data/raw/
  git add .
  git commit -m "Versioned raw data"
  ```

## Model Training & Experiment Tracking
- **Training Process:** `src/models/train.py` handles training using TensorFlow/Keras.
- **Evaluation Metrics:** RMSE, MAE, and R² are computed using scikit-learn.
- **Experiment Tracking:** MLflow logs experiments, model performance, and hyperparameters.
  ```sh
  mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```

## API Deployment & Monitoring
- **FastAPI Server:** Hosted using Uvicorn (`src/api/app.py`).
- **Monitoring with Prometheus & Grafana:**
  - `prometheus_client` collects API performance and model prediction metrics.
  - Grafana visualizes real-time performance.
- **Metrics Tracked:**
  - Request Latency (Histograms)
  - API Call Counts (Counters)
  - Model Prediction Distributions (Gauges)

## Data Processing & Visualization
- **Data Handling:** `pandas` and `numpy` are used for data transformation.
- **Visualization:** `matplotlib` and `seaborn` generate performance and analysis plots.
- **Automated Logging:** Python’s `logging` module captures events, warnings, and errors.

## Achievements & Project Outcomes
- **End-to-End MLOps Pipeline:** Covers data ingestion, versioning, model training, and deployment.
- **Reproducibility:** Experiment tracking via MLflow.
- **Monitoring & Performance Tracking:** Real-time API metrics via Prometheus and Grafana.
- **Automated Data Management:** Scheduled data collection and versioning ensure continuous improvement.

---
**Author:** Muhammad Waleed
**License:** MIT License  
**Contact:** waleedyaseen444@gmail.com
