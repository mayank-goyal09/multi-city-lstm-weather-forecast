import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

# =========================
# 1. Model architecture 🧠
# =========================

FEATURE_COLS = ["temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m"]
TARGET_COL = "temperature_2m"
LOOKBACK_HOURS = 30 * 24   # 720
HORIZON_HOURS = 7 * 24     # 168

def build_lstm_model(timesteps, num_features, horizon):
    model = models.Sequential([
        layers.Input(shape=(timesteps, num_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dense(horizon)  # 168 outputs
    ])
    return model

@st.cache_resource
def load_lstm_model():
    timesteps = LOOKBACK_HOURS
    num_features = len(FEATURE_COLS)
    horizon = HORIZON_HOURS

    model = build_lstm_model(timesteps, num_features, horizon)
    # Load weights saved as weather_lstm_7day.weights.h5
    model.load_weights("weather_lstm_7day.weights.h5")
    model.compile(optimizer="adam", loss="mse")
    return model

model = load_lstm_model()

# =========================
# 2. Data loading & scalers 📊
# =========================

@st.cache_data
def load_history():
    df = pd.read_csv("weather_hourly_history_openmeteo.csv", parse_dates=["time"])
    df = df.sort_values(["city", "time"]).reset_index(drop=True)
    return df

history_df = load_history()

@st.cache_resource
def build_city_scalers(df):
    scalers = {}
    for c in df["city"].unique():
        city_df = df[df["city"] == c]
        scaler = MinMaxScaler()
        scaler.fit(city_df[FEATURE_COLS])
        scalers[c] = scaler
    return scalers

scalers = build_city_scalers(history_df)

def inverse_temp(scaled_temp_1d, scaler):
    """
    Inverse-transform temperature_2m only, using a multi-feature scaler.
    """
    dummy = np.zeros((len(scaled_temp_1d), len(FEATURE_COLS)))
    dummy[:, FEATURE_COLS.index("temperature_2m")] = scaled_temp_1d
    inv = scaler.inverse_transform(dummy)
    return inv[:, FEATURE_COLS.index("temperature_2m")]

# =========================
# 3. Window + forecast utils 🔁
# =========================

def build_last_window_for_city(df, city, lookback_hours):
    """
    Returns:
      X_sample: (1, lookback, num_features)
      scaler: city scaler
      last_time: last timestamp in history
    """
    city_df = df[df["city"] == city].sort_values("time")
    if len(city_df) < lookback_hours:
        raise ValueError(f"Not enough history for {city}. Need {lookback_hours} hours.")

    city_df_hist = city_df.tail(lookback_hours)
    scaler = scalers[city]
    feats_scaled = scaler.transform(city_df_hist[FEATURE_COLS])
    X = feats_scaled[np.newaxis, :, :]   # (1, lookback, features)
    last_time = city_df_hist["time"].iloc[-1]
    return X, scaler, last_time

def forecast_7_days(model, X_sample, scaler, last_time):
    """
    Predict 168h forecast and return:
      df_full: full hourly forecast
      df_display: Day 1 hourly + Days 2–7 every 4h
    """
    y_pred = model.predict(X_sample)[0]         # (168,)
    y_pred_c = inverse_temp(y_pred, scaler)    # back to °C

    hours = np.arange(HORIZON_HOURS)
    # Future timestamps: start 1 hour after last history time
    times_future = [last_time + timedelta(hours=int(h+1)) for h in hours]

    df_full = pd.DataFrame({
        "time": times_future,
        "hour_ahead": hours,
        "pred_temp_c": y_pred_c
    })

    # Display subset: Day 1 hourly + 4-hourly for Days 2–7
    mask_display = (df_full["hour_ahead"] < 24) | (
        (df_full["hour_ahead"] >= 24) &
        (((df_full["hour_ahead"] - 24) % 4) == 0)
    )
    df_display = df_full[mask_display].reset_index(drop=True)
    return df_full, df_display

# =========================
# 4. Streamlit UI 🎛️
# =========================

st.set_page_config(
    page_title="7-Day Weather Forecast (LSTM)",
    layout="wide",
)

st.title("🌦️ 7-Day Weather Forecast with LSTM")
st.markdown(
    """
This app uses a **Recurrent Neural Network (LSTM)** trained on **Open-Meteo hourly data**  
for four cities (Delhi, Mumbai, New York, Los Angeles) to forecast the next **7 days of temperature**.

- History window: **30 days of hourly data (720 hours)**
- Forecast horizon: **7 days of hourly data (168 hours)**
- Display:
  - **Day 1**: hourly forecast
  - **Days 2–7**: quarter-day forecast (every 4 hours)
"""
)

city_options = sorted(history_df["city"].unique())
city = st.selectbox("Choose a city", city_options, index=0)

if st.button("Generate 7-Day Forecast"):
    try:
        with st.spinner("Preparing data and running LSTM model..."):
            X_sample, scaler, last_time = build_last_window_for_city(
                history_df, city, LOOKBACK_HOURS
            )
            df_full, df_display = forecast_7_days(model, X_sample, scaler, last_time)

        # Layout: left = chart, right = table
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"7-Day Hourly Forecast for {city.capitalize()} (°C)")
            chart_df = df_full.set_index("time")[["pred_temp_c"]]
            st.line_chart(chart_df)

        with col2:
            st.subheader("Day 1 Hourly + Days 2–7 (4-hour steps)")
            st.dataframe(df_display)

        st.caption(
            f"Last historical timestamp used: {last_time} | "
            "Forecast is model output, not actual observed data."
        )

    except Exception as e:
        st.error(f"Error generating forecast: {e}")
else:
    st.info("Select a city and click **Generate 7-Day Forecast** to see predictions.")
