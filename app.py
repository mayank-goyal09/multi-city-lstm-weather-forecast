import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

# =========================
# Page Config - MUST BE FIRST
# =========================
st.set_page_config(
    page_title="🌦️ WeatherLens AI | 7-Day Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Premium Custom CSS Styling
# =========================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    /* Root Variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --weather-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --sunset-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --night-gradient: linear-gradient(135deg, #0c1445 0%, #1a237e 50%, #311b92 100%);
        --card-bg: rgba(255, 255, 255, 0.08);
        --glass-border: rgba(255, 255, 255, 0.18);
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --accent-blue: #4facfe;
        --accent-purple: #667eea;
        --accent-pink: #f093fb;
        --accent-orange: #fee140;
    }
    
    /* Global Body Styling */
    .stApp {
        background: var(--night-gradient);
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated Background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(ellipse at 20% 20%, rgba(102, 126, 234, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(240, 147, 251, 0.12) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(79, 172, 254, 0.08) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
        animation: shimmer 15s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; }
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main .block-container {
        padding: 2rem 3rem !important;
        max-width: 1400px;
    }
    
    /* Hero Header */
    .hero-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(240, 147, 251, 0.15) 100%);
        border-radius: 24px;
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(79, 172, 254, 0.1), transparent 30%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #4facfe 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: var(--text-secondary);
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--weather-gradient);
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #0c1445;
        margin-top: 1.5rem;
        position: relative;
        z-index: 1;
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.4);
    }
    
    /* Glass Cards */
    .glass-card {
        background: var(--card-bg);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.75rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
        border-color: rgba(255, 255, 255, 0.25);
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(102, 126, 234, 0.1) 100%);
        border: 1px solid rgba(79, 172, 254, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: scale(1.03);
        border-color: rgba(79, 172, 254, 0.4);
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.2);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    
    /* City Selector Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: white !important;
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--accent-blue) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: var(--weather-gradient) !important;
        color: #0c1445 !important;
        border: none !important;
        padding: 0.875rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 6px 25px rgba(79, 172, 254, 0.4) !important;
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 35px rgba(79, 172, 254, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98) !important;
    }
    
    /* Info, Spinner & Alert Styling */
    .stInfo, .stWarning, .stError, .stSuccess {
        background: rgba(79, 172, 254, 0.1) !important;
        border: 1px solid rgba(79, 172, 254, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stSpinner > div {
        border-color: var(--accent-blue) transparent transparent transparent !important;
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(10px);
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] > div {
        background: transparent !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(240, 147, 251, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(12px);
    }
    
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.25rem;
        font-weight: 700;
        background: var(--weather-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
        font-weight: 500;
    }
    
    /* Weather Icons Animation */
    .weather-icon-float {
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .section-header::after {
        content: '';
        flex: 1;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), transparent);
        border-radius: 1px;
    }
    
    /* Footer Styling */
    .footer {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        backdrop-filter: blur(16px);
    }
    
    .footer-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    
    .social-links {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    
    .social-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 20px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    .social-btn.linkedin {
        background: linear-gradient(135deg, #0077B5 0%, #00a0dc 100%);
        color: white;
    }
    
    .social-btn.github {
        background: linear-gradient(135deg, #24292e 0%, #4d5560 100%);
        color: white;
    }
    
    .social-btn.portfolio {
        background: var(--primary-gradient);
        color: white;
    }
    
    .social-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .footer-credit {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    
    .footer-credit span {
        color: var(--accent-pink);
    }
    
    /* City Card */
    .city-card {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.2) 0%, rgba(0, 242, 254, 0.1) 100%);
        border: 1px solid rgba(79, 172, 254, 0.25);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .city-name {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Pulse Animation */
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Temperature Display */
    .temp-display {
        font-family: 'Outfit', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        .main .block-container {
            padding: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

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
    dummy = np.zeros((len(scaled_temp_1d), len(FEATURE_COLS)))
    dummy[:, FEATURE_COLS.index("temperature_2m")] = scaled_temp_1d
    inv = scaler.inverse_transform(dummy)
    return inv[:, FEATURE_COLS.index("temperature_2m")]

# =========================
# 3. Window + forecast utils 🔁
# =========================

def build_last_window_for_city(df, city, lookback_hours):
    city_df = df[df["city"] == city].sort_values("time")
    if len(city_df) < lookback_hours:
        raise ValueError(f"Not enough history for {city}. Need {lookback_hours} hours.")

    city_df_hist = city_df.tail(lookback_hours)
    scaler = scalers[city]
    feats_scaled = scaler.transform(city_df_hist[FEATURE_COLS])
    X = feats_scaled[np.newaxis, :, :]
    last_time = city_df_hist["time"].iloc[-1]
    return X, scaler, last_time

def forecast_7_days(model, X_sample, scaler, last_time):
    y_pred = model.predict(X_sample, verbose=0)[0]
    y_pred_c = inverse_temp(y_pred, scaler)

    hours = np.arange(HORIZON_HOURS)
    times_future = [last_time + timedelta(hours=int(h+1)) for h in hours]

    df_full = pd.DataFrame({
        "time": times_future,
        "hour_ahead": hours,
        "pred_temp_c": y_pred_c
    })

    mask_display = (df_full["hour_ahead"] < 24) | (
        (df_full["hour_ahead"] >= 24) &
        (((df_full["hour_ahead"] - 24) % 4) == 0)
    )
    df_display = df_full[mask_display].reset_index(drop=True)
    return df_full, df_display

# =========================
# City Icons Mapping
# =========================
CITY_ICONS = {
    "delhi": "🏛️",
    "mumbai": "🌊",
    "new york": "🗽",
    "los angeles": "🌴",
    "new_york": "🗽",
    "los_angeles": "🌴"
}

def get_city_icon(city):
    return CITY_ICONS.get(city.lower().replace(" ", "_"), "🏙️")

def get_temp_emoji(temp):
    if temp < 0:
        return "❄️"
    elif temp < 10:
        return "🥶"
    elif temp < 20:
        return "🌤️"
    elif temp < 30:
        return "☀️"
    elif temp < 40:
        return "🔥"
    else:
        return "🌡️"

# =========================
# 4. Streamlit UI 🎛️
# =========================

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🌤️ WeatherLens AI</div>
    <div class="hero-subtitle">Advanced 7-Day Weather Forecasting powered by Deep Learning</div>
    <div class="hero-badge">
        <span>🧠</span>
        <span>LSTM Neural Network</span>
        <span>•</span>
        <span>📊</span>
        <span>168-Hour Predictions</span>
    </div>
</div>
""", unsafe_allow_html=True)
