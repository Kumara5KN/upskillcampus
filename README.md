🚦 Smart City Traffic Forecasting System

Project Overview

This project uses Machine Learning (XGBoost) to forecast hourly traffic volume across four city junctions. The goal is to help city teams proactively manage congestion and optimize traffic lights.

🎯 Goal

The main objective is to predict hourly vehicle count for the next four months, specifically accounting for:

Daily and weekly traffic patterns.

The major traffic drop caused by Public Holidays.

💻 Technology Stack

Component

Purpose

Model

XGBoost Regressor (High-accuracy forecasting)

Framework

Streamlit (app.py) for the web application

Language

Python 3.9+

Dependencies

pandas, numpy, joblib, plotly, xgboost

⚙️ Setup and Installation

1. Prerequisites

Python 3.9+ is required.

2. Clone the Repository

git clone [PASTE_YOUR_REPO_URL_HERE]
cd smart-city-traffic-forecaster


3. Install Dependencies

Install all required libraries using requirements.txt:

pip install -r requirements.txt


4. Required Files

The following essential files must be in the root directory:

app.py (The application)

xgb_traffic_model.joblib (The trained model)

model_features.joblib (The 18 required feature list)

requirements.txt

train_aWnotuB.csv / datasets_8494_11879_test_BdBKkAj.csv (Source data)

▶️ How to Run the Application

Run the app from your terminal:

streamlit run app.py


The application will open in your browser for you to generate forecasts.

📊 Model Performance & Features

Key Metric: Root Mean Square Error ($\text{RMSE}$) $\approx 8.5$ vehicles. This demonstrates high precision on unseen data.

Most Important Features: Junction ID and Hour of Day were ranked highest, confirming the model effectively captures spatial and time-of-day patterns.
