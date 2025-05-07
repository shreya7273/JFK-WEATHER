import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

print("JFK Weather Script Started")

# === PHASE 1: LOAD & CLEAN DATA ===
df = pd.read_csv('jfk_weather_cleaned.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

print("Loaded data:", df.shape)
print(df.head())

#Part 2: INDIVIDUAL EDA VISUALS
os.makedirs("visuals", exist_ok=True)

# 1. Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("visuals/correlation_matrix.png")
plt.show()
plt.close()

# 2. Temperature Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['HOURLYDRYBULBTEMPF'], bins=30, kde=True, color='orange')
plt.title("Dry Bulb Temperature Distribution")
plt.xlabel("Temperature (°F)")
plt.savefig("visuals/temp_distribution.png")
plt.show()
plt.close()

# 3. Visibility Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['HOURLYVISIBILITY'], bins=30, kde=True, color='skyblue')
plt.title("Visibility Distribution")
plt.xlabel("Visibility (miles)")
plt.savefig("visuals/visibility_distribution.png")
plt.show()
plt.close()

# 4. Humidity vs Temperature
plt.figure(figsize=(8, 5))
sns.scatterplot(x='HOURLYDRYBULBTEMPF', y='HOURLYRelativeHumidity', data=df)
plt.title("Humidity vs Temperature")
plt.xlabel("Temperature (°F)")
plt.ylabel("Relative Humidity (%)")
plt.savefig("visuals/humidity_vs_temp.png")
plt.show()
plt.close()


# Part 3: FEATURE SCALING
features = df.drop(columns=['HOURLYPrecip'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
df_scaled = pd.DataFrame(scaled_features, columns=features.columns, index=df.index)
df_scaled['HOURLYPrecip'] = df['HOURLYPrecip']
print("Feature scaling complete.")

# part 4: RANDOM FOREST REGRESSION
X = df_scaled.drop(columns=['HOURLYDRYBULBTEMPF'])
y = df_scaled['HOURLYDRYBULBTEMPF']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("\n Random Forest Results:")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

# part 5: ARIMA FORECASTING
daily_temp = df['HOURLYDRYBULBTEMPF'].resample('D').mean()
model_arima = ARIMA(daily_temp, order=(5, 1, 0))
fit_arima = model_arima.fit()
forecast = fit_arima.forecast(steps=30)

plt.figure(figsize=(10, 5))
daily_temp[-100:].plot(label='Historical')
forecast.plot(label='Forecast', color='red')
plt.title("ARIMA Forecast - Temperature (Next 30 Days)")
plt.legend()
plt.savefig("visuals/arima_forecast_temp.png")
plt.close()
print("ARIMA forecast saved: visuals/arima_forecast_temp.png")

# part 6: LSTM FORECASTING (PRECIPITATION)
precip = df[['HOURLYPrecip']].values
minmax = MinMaxScaler()
precip_scaled = minmax.fit_transform(precip)

X_seq, y_seq = [], []
for i in range(60, len(precip_scaled)):
    X_seq.append(precip_scaled[i-60:i])
    y_seq.append(precip_scaled[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

model_lstm = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_seq.shape[1], 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_seq, y_seq, epochs=5, batch_size=64, verbose=1)

last_seq = X_seq[-1].reshape(1, 60, 1)
next_precip = model_lstm.predict(last_seq)
next_precip = minmax.inverse_transform(next_precip)
print("LSTM Predicted Precipitation (Next Step):", round(next_precip[0][0], 4))

# part 7: K-MEANS CLUSTERING
X_cluster = df_scaled.drop(columns=['HOURLYPrecip'])
kmeans = KMeans(n_clusters=3, random_state=42)
df_scaled['WeatherCluster'] = kmeans.fit_predict(X_cluster)

sns.scatterplot(data=df_scaled, x='HOURLYDRYBULBTEMPF', y='HOURLYRelativeHumidity',
                hue='WeatherCluster', palette='tab10')
plt.title("Clustered Weather Patterns")
plt.savefig("visuals/weather_clusters.png")
plt.close()
print("K-Means clustering complete and visual saved.")

# part 8: summery
print("\nALL PHASES COMPLETE!")
print("EDA saved to: visuals/eda_summary_plot.png")
print("Random Forest: MAE =", round(mae, 4), "| RMSE =", round(rmse, 4))
print("ARIMA forecast saved")
print("LSTM precipitation =", round(next_precip[0][0], 4))
print("K-Means clusters saved to: visuals/weather_clusters.png")
