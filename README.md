🚦 Smart City Traffic Forecasting System

Project Overview

This project implements a Machine Learning solution to accurately forecast hourly traffic volume across four major city junctions. The goal is to provide municipal planning and traffic control teams with proactive intelligence to mitigate congestion and optimize signal timing, directly supporting the Smart City initiative.

🎯 Goal

The core objective is to predict the hourly vehicle count for the next four months on four specific road junctions, successfully modeling complex time-series patterns, including:

Daily and weekly seasonality (rush hour peaks).

The significant impact of Public Holidays, which drastically alter normal traffic flow.

💻 Technology Stack

Component

Purpose

Model

XGBoost Regressor (for high-accuracy time-series regression)

Framework

Streamlit (app.py) for a functional, interactive web application

Language

Python 3.9+

Dependencies

pandas, numpy, joblib, plotly, xgboost

⚙️ Setup and Installation

1. Prerequisites

Ensure you have Python 3.9+ installed.

2. Clone the Repository

git clone [PASTE_YOUR_REPO_URL_HERE]
cd smart-city-traffic-forecaster


3. Install Dependencies

Install all required libraries using the provided requirements.txt file:

pip install -r requirements.txt


4. Required Files

Ensure the following files are present in the root directory:

app.py (The main Streamlit application)

xgb_traffic_model.joblib (The trained XGBoost model)

model_features.joblib (The list of 18 features required by the model)

requirements.txt

train_aWnotuB.csv / datasets_8494_11879_test_BdBKkAj.csv (Source data)

▶️ How to Run the Application

Once the dependencies are installed, run the Streamlit application from your terminal:

streamlit run app.py


The application will automatically open in your web browser, allowing you to select a future date, time, and junction to generate a traffic forecast.

📊 Model Performance & Features

The model was trained on historical data from 2015-2017.

Key Metric: Root Mean Square Error ($\text{RMSE}$) $\approx 8.5$ vehicles on the unseen test set, demonstrating high precision.

Most Important Features: The model validated the feature engineering by ranking Junction ID and Hour of Day as the strongest predictors, confirming its ability to capture both spatial and temporal variance.
