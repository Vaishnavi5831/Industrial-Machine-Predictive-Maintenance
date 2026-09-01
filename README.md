# Industrial Machine Intelligence & Predictive Maintenance

##  Project Overview

This project focuses on developing an intelligent predictive maintenance system for industrial turbofan engines using Machine Learning and Data Analytics.

The system analyzes engine sensor data to monitor machine health, detect abnormal behavior, predict failures, estimate Remaining Useful Life (RUL), and support predictive maintenance decisions.

---

##  Dataset

**NASA C-MAPSS-1 Turbofan Engine Degradation Dataset**

The dataset contains run-to-failure sensor measurements from turbofan engines under different operating conditions.

### Dataset Features

- Engine ID
- Operating Cycle
- Operating Settings
- Multiple Sensor Measurements
- Engine Degradation Information

---

## Project Objectives

- Monitor industrial machine health
- Analyze sensor behavior
- Detect abnormal machine conditions
- Predict possible machine failures
- Classify failure conditions
- Predict Remaining Useful Life (RUL)
- Perform time-series analysis
- Build machine learning models
- Provide model explainability
- Integrate ML models with an API
- Develop an interactive monitoring dashboard

---

#  Project Modules

## 1. Data Engineering & Preprocessing

This module prepares the raw NASA C-MAPSS dataset for machine learning.

### Tasks

- Load NASA C-MAPSS dataset
- Data cleaning
- Missing-value handling
- Duplicate checking
- Feature selection
- Data transformation
- Normalization / scaling
- Train-test preparation

---

## 2. Sensor Data Analysis

This module analyzes engine sensor measurements to understand machine behavior and degradation.

### Analysis

- Sensor trends
- Sensor distributions
- Sensor correlations
- Engine degradation patterns
- Important sensors

### Visualizations

- Line plots
- Histograms
- Box plots
- Correlation heatmaps
- Sensor trend charts

---

## 3. Machine Failure Prediction

Machine Learning models are used to predict whether an engine is approaching a failure condition.

### Models

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 4. Failure Type Classification

This module identifies different failure or degradation conditions using machine learning classification techniques.

### Techniques

- Feature engineering
- Classification models
- Model training
- Model evaluation
- Confusion matrix
- Classification report

---

## 5. Remaining Useful Life (RUL) Prediction

The system estimates the remaining operating cycles before an engine reaches its failure condition.

### Models

- Linear Regression
- Random Forest Regressor
- Gradient Boosting
- XGBoost Regressor

### Evaluation Metrics

- MAE
- RMSE
- R² Score

---

## 6. Anomaly Detection

This module detects unusual or abnormal engine behavior from sensor measurements.

### Techniques

- Isolation Forest
- Statistical analysis
- Sensor threshold analysis

### Output

- Normal Condition
- Anomalous Condition

---

## 7. Time-Series Analysis

This module analyzes sensor values over operating cycles to understand degradation patterns.

### Analysis Includes

- Degradation trends
- Sensor behavior
- Performance changes
- Failure progression
- Operating cycle analysis

---

## 8. Feature Engineering & Model Explainability

Important features are created from raw sensor data to improve machine learning performance.

### Feature Engineering

- Rolling statistics
- Lag features
- Sensor trends
- Moving averages
- Feature selection

### Model Explainability

- Feature importance
- SHAP
- Model interpretation
- Important sensor identification

---

## 9. ML Model / API Integration

The trained machine learning models are integrated with an API for prediction.

### Technologies

- Python
- FastAPI
- Scikit-learn
- Joblib

### API Predictions

The API can provide:

- Failure prediction
- RUL prediction
- Anomaly status
- Machine health status

---

#  10. Monitoring & ML Dashboard

An interactive monitoring dashboard is developed using **Streamlit**.

### Dashboard Sections

- Overview
- Engine Selection
- Machine Health
- Sensor Analysis
- Failure Prediction
- Failure Classification
- RUL Prediction
- Anomaly Detection
- Time-Series Monitoring
- Model Performance
- API Prediction

---

# Live Dashboard

The project includes an interactive AI-based engine health monitoring dashboard built using Streamlit.

🔗 **Live Dashboard:**  
https://restore-ware-knives-nail.trycloudflare.com

### Dashboard Features

- Engine Selection
- Engine Health Monitoring
- Current Cycle Tracking
- Machine Status
- Failure Prediction
- RUL Prediction
- Sensor Analysis
- Anomaly Detection
- Time-Series Monitoring
- Interactive Visualizations

---

#  Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Explainable AI

- SHAP

### API

- FastAPI
- Uvicorn

### Dashboard

- Streamlit

### Model Management

- Joblib

### Development Environment

- Jupyter Notebook
- Git
- GitHub

---

# Project Structure

```text
Industrial-Machine-Intelligence-Predictive-Maintenance/
│
├── data/
│   └── NASA_CMAPSS/
│
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Sensor_Data_Analysis.ipynb
│   ├── 03_Failure_Prediction.ipynb
│   ├── 04_Failure_Classification.ipynb
│   ├── 05_RUL_Prediction.ipynb
│   ├── 06_Anomaly_Detection.ipynb
│   ├── 07_Time_Series_Analysis.ipynb
│   └── 08_Feature_Engineering.ipynb
│
├── models/
│   └── trained_models/
│
├── api/
│   └── app.py
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
│
└── README.md
## Author
Vaishnavi
