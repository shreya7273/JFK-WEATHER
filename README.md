JFK-WEATHER

A data science project that analyzes and forecasts weather conditions at JFK Airport using historical weather data, machine learning models, and time series analysis.

Project Overview

This project includes:

Cleaning and preprocessing JFK Airport weather data
Exploratory Data Analysis (EDA)
Clustering weather patterns (KMeans & Hierarchical)
Forecasting using:
  Random Forest Regressor
  ARIMA
  LSTM

Folder Structure
Jfk_weather_report

Jfk_Weather.ipynb          #Jupyter Notebook
visuals/                   # Plots and charts from EDA and models
jfk_weather_main.py        # code
jfk_weather_cleaned.csv    # Cleaned dataset
README.md                  # Project documentation
requirements.txt           # Python dependencies


1. Clone the repository
   git clone https://github.com/shreya7273/JFK-WEATHER.git
   cd JFK-WEATHER


2. Create virtual environment (optional)
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Run main script
   python scripts/main.py

Features

Clean and preprocess historical weather data
Visualizations: correlation matrix, line charts, seasonal trends
Weather pattern clustering
Forecasting models: Random Forest, ARIMA, and LSTM

Output

Visualizations are saved in the visuals folder after running the scripts:
Temperature & humidity trends
PCA cluster plots
Forecasted values

Improvements Planned

Real-time API integration
Model optimization
Web interface for predictions

