import requests
import pandas as pd
from datetime import date, timedelta

# ============================
# 1. City configuration 🌍
# ============================

CITIES = {
    "delhi":       {"lat": 28.6139, "lon": 77.2090},
    "mumbai":      {"lat": 19.0760, "lon": 72.8777},
    "new_york":    {"lat": 40.7128, "lon": -74.0060},
    "los_angeles": {"lat": 34.0522, "lon": -118.2437},
}

# ============================
# 2. Fetch hourly forecast (for testing only) ⏱️
# ============================

def fetch_hourly_forecast(lat, lon, forecast_days=7, timezone="auto"):
    """
    Fetch hourly forecast data (up to several days ahead) from Open-Meteo.
    Useful to quickly inspect structure; not used for training history.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
        ],
        "forecast_days": forecast_days,
        "timezone": timezone,
        "models": "best_match",  # Let Open-Meteo auto-select model
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    hourly = pd.DataFrame(data["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly.set_index("time", inplace=True)
    return hourly

# ============================
# 3. Fetch historical hourly data (for training) 📜
# ============================

def fetch_hourly_history(lat, lon, start_date, end_date, timezone="auto"):
    """
    Fetch historical hourly weather data from Open-Meteo Historical Weather API.
    Dates are strings in 'YYYY-MM-DD' format.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"  # Historical API
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
        ],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    hourly = pd.DataFrame(data["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly.set_index("time", inplace=True)
    return hourly

# ============================
# 4. Fetch for all cities and save CSV 💾
# ============================

def fetch_all_cities_history(cities, days=365):
    """
    Fetch historical hourly weather for multiple cities and combine into one DataFrame.
    """
    end = date.today()
    start = end - timedelta(days=days)

    all_frames = []
    for name, loc in cities.items():
        print(f"Fetching {days} days of history for {name}...")
        df_city = fetch_hourly_history(
            lat=loc["lat"],
            lon=loc["lon"],
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
        df_city["city"] = name
        all_frames.append(df_city)

    full_df = pd.concat(all_frames)
    full_df.reset_index(inplace=True)
    full_df.rename(columns={"index": "time"}, inplace=True)
    return full_df


if __name__ == "__main__":
    # 1) Quick test: see one city forecast
    # delhi_forecast = fetch_hourly_forecast(**CITIES["delhi"])
    # print(delhi_forecast.head())

    # 2) Main: fetch history and save
    history_df = fetch_all_cities_history(CITIES, days=365)
    history_df.to_csv("weather_hourly_history_openmeteo.csv", index=False)
    print("✅ Saved: weather_hourly_history_openmeteo.csv")
