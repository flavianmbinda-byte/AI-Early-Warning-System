import numpy as np
import pandas as pd


# =========================
# VPD FUNCTION
# =========================
def compute_vpd(df):

    df = df.copy()

    df["RH"] = df["RH"].clip(0, 100)

    df["TMEAN"] = (
        df["TMIN"] + df["TMAX"]
    ) / 2

    es = 0.6108 * np.exp(
        (17.27 * df["TMEAN"]) /
        (df["TMEAN"] + 237.3)
    )

    ea = es * (
        df["RH"] / 100
    )

    df["VPD"] = es - ea

    df["VPD"] = df["VPD"].clip(lower=0)

    return df


# =========================
# FAW FUNCTION
# =========================
def faw_risk(temp, rh, rain):

    if pd.isna(temp) or pd.isna(rh) or pd.isna(rain):
        return "Low"

    score = 0

    if 25 <= temp <= 32:
        score += 2
    elif 20 <= temp < 25:
        score += 1

    if rh >= 60:
        score += 2
    elif rh >= 40:
        score += 1

    if 50 <= rain <= 150:
        score += 2
    elif rain >= 20:
        score += 1

    if score >= 5:
        return "High"
    elif score >= 3:
        return "Medium"
    else:
        return "Low"


# =========================
# AI CROP HEALTH
# =========================
def calculate_crop_health(weather, vpd, faw):

    rainfall = weather["rainfall"]

    if rainfall >= 5:
        rain_score = 100
    elif rainfall >= 2:
        rain_score = 75
    elif rainfall > 0:
        rain_score = 50
    else:
        rain_score = 25

    temp = weather["temperature"]

    if 20 <= temp <= 30:
        temp_score = 100
    elif 15 <= temp < 20 or 30 < temp <= 35:
        temp_score = 70
    else:
        temp_score = 40

    if vpd < 0.8:
        water_score = 100
    elif vpd < 1.2:
        water_score = 75
    elif vpd < 1.6:
        water_score = 50
    else:
        water_score = 25

    if faw == "Low":
        faw_score = 100
    elif faw == "Medium":
        faw_score = 70
    else:
        faw_score = 30

    overall = round(
        (rain_score + temp_score + water_score + faw_score) / 4
    )

    return {
        "rain": rain_score,
        "temperature": temp_score,
        "water": water_score,
        "faw": faw_score,
        "overall": overall
    }


# =========================
# CLIMATE ADJUSTMENT
# =========================
def climate_adjustment(weather, vpd):

    score = 0

    if weather["rainfall"] >= 10:
        score += 3
    elif weather["rainfall"] >= 5:
        score += 2
    elif weather["rainfall"] >= 1:
        score += 1
    else:
        score -= 3

    if weather["humidity"] >= 80:
        score += 2
    elif weather["humidity"] >= 60:
        score += 1
    elif weather["humidity"] < 40:
        score -= 2

    if weather["temperature"] >= 35:
        score -= 2
    elif weather["temperature"] >= 32:
        score -= 1

    if vpd >= 2.5:
        score -= 3
    elif vpd >= 2.0:
        score -= 2
    elif vpd >= 1.5:
        score -= 1
    elif vpd < 1.0:
        score += 1

    return max(-5, min(score, 5))