# Industrial Machine Intelligence & Predictive Maintenance

## Project Overview

This project focuses on developing an intelligent predictive maintenance system for industrial turbofan engines using Machine Learning and Data Analytics.

The system analyzes engine sensor data to monitor machine health, detect abnormal behavior, predict failures, estimate Remaining Useful Life (RUL), and support predictive maintenance decisions.

---

## 📊 Dataset

**NASA C-MAPSS-1 Turbofan Engine Degradation Dataset**

The dataset contains run-to-failure sensor measurements from turbofan engines under different operating conditions.

### Dataset Features

- Engine ID
- Operating Cycle
- Operating Settings
- Multiple Sensor Measurements
- Engine degradation information

---

## 🎯 Project Objectives

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

- Load NASA C-MAPSS dataset
- Data cleaning
- Missing-value handling
- Feature selection
- Data transformation
- Normalization / scaling
- Train-test preparation

---

## 2. Sensor Data Analysis

Analyze sensor measurements to understand:

- Sensor trends
- Sensor distributions
- Correlations
- Engine degradation patterns
- Important sensors

Visualizations include:

- Line plots
- Histograms
- Box plots
- Correlation heatmaps

---

## 3. Machine Failure Prediction

Machine Learning models are used to predict whether an engine is approaching a failure condition.

Possible models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 4. Failure Type Classification

This module identifies different failure/degradation conditions using machine learning classification techniques.

Techniques include:

- Feature engineering
- Classification models
- Model evaluation
- Confusion matrix
- Classification report

---

## 5. Remaining Useful Life (RUL) Prediction

The system estimates how many operating cycles remain before the engine reaches its failure condition.

Models can include:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting
- XGBoost Regressor

Evaluation metrics:

- MAE
- RMSE
- R² Score

---

## 6. Anomaly Detection

Detect unusual engine behavior from sensor measurements.

Techniques:

- Isolation Forest
- Statistical analysis
- Sensor threshold analysis

Output:

- Normal condition
- Anomalous condition

---

## 7. Time-Series Analysis

Analyze engine sensor values over operating cycles.

This helps identify:

- Degradation trends
- Sensor behavior
- Performance changes
- Failure progression

---

## 8. Feature Engineering & Model Explainability

Important features are created from raw sensor data.

Techniques include:

- Rolling statistics
- Lag features
- Sensor trends
- Feature selection

Model explainability can be performed using:

- Feature importance
- SHAP
- Model interpretation

---

## 9. ML Model / API Integration

The trained machine learning model is integrated with an API.

Possible technology:

- Python
- FastAPI
- Scikit-learn
- Joblib

The API can receive sensor information and return:

- Failure prediction
- RUL prediction
- Anomaly status

---

## 10. Monitoring & ML Dashboard

An interactive dashboard is developed using **Streamlit**.

### Dashboard Sections

- Overview
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

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- FastAPI
- Streamlit
- Joblib
- Jupyter Notebook

---

# 📁 Project Structure

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
