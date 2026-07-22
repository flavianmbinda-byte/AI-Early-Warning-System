import os
import requests
import streamlit as st

try:
    API_KEY = st.secrets["OPENWEATHER_API_KEY"]
except Exception:
    API_KEY = os.getenv("OPENWEATHER_API_KEY")

LOCATIONS = {
    "Mbeya": {
        "lat": -8.9094,
        "lon": 33.4608
    },

    "Kongwa": {
        "lat": -6.2000,
        "lon": 36.4167
    },

    "Zanzibar": {
        "lat": -6.1659,
        "lon": 39.2026
    }
}


def get_weather(region):

    lat = LOCATIONS[region]["lat"]
    lon = LOCATIONS[region]["lon"]

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}"
        f"&lon={lon}"
        f"&appid={API_KEY}"
        f"&units=metric"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()

    weather = {

        "temperature": data["main"]["temp"],
        "temp_max": data["main"]["temp_max"],
        "temp_min": data["main"]["temp_min"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind": data["wind"]["speed"],
        "rainfall": 0
    }

    if "rain" in data:
        weather["rainfall"] = data["rain"].get("1h", 0)

    return weather