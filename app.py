import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import requests
from weather import get_weather
from database import conn, cursor, save_weather
from utils import rainfall_state, ai_image
from markov import (
    build_transition_matrix,
    predict_next
)
from climate import (
    compute_vpd,
    faw_risk,
    calculate_crop_health,
    climate_adjustment
)
from forecast import (
    adjust_monthly_rain,
    translate_rain,
    translate_faw,
    translate_vpd
)
from yield_model import (
    load_block_data,
    prepare_annual_features,
    get_region_features,
    get_models,
    prepare_training_data,
    train_models,
    predict_yield
)
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Early Warning System",
    page_icon="images/logo.png",
    layout="wide"
)
# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

/* Images */
img {
    border-radius: 15px;
    width: 100%;
    height: auto;
    object-fit: cover;
}

/* Cards */
.card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 15px;
    text-align: center;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-3px);
}

/* Titles */
.title {
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}

/* States */
.state {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: green;
}

/* Mobile responsiveness */
@media (max-width: 768px) {

    .title {
        font-size: 17px;
    }

    .state {
        font-size: 24px;
    }

    .card {
        padding: 12px;
        border-radius: 15px;
    }

    img {
        border-radius: 12px;
    }
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f5f7fa;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-weight: bold;
}

/* Success boxes */
.stSuccess {
    border-radius: 12px;
}

/* Warning boxes */
.stWarning {
    border-radius: 12px;
}

/* Error boxes */
.stError {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LANGUAGE
# =========================
language = st.sidebar.selectbox(
    "🌍 Language / Lugha",
    ["English", "Kiswahili"]
)

def T(en, sw):
    return en if language == "English" else sw
# =========================
# HEADER
# =========================

col1, col2, col3 = st.columns([3,1,3])

with col2:

    st.image(
        "images/logo.png",
        width=90
    )

st.markdown(
    f"""
    <h1 style='text-align:center;'>

    {T(
        "🌽 AI Early Warning System",
        "🌽 Mfumo wa AI wa Tahadhari ya Mapema"
    )}

    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <p style='text-align:center;
              font-size:16px;
              color:gray;
              max-width:700px;
              margin:auto;'>

    {T(
        "AI-powered climate, pest, and yield forecasting to support informed farming decisions in Tanzania.",
        "Mfumo wa Akili Unde (AI) wa kutabiri hali ya hewa, wadudu waharibifu, na mavuno kwa ajili ya kusaidia maamuzi sahihi ya kilimo nchini Tanzania."
    )}

    </p>
    """,
    unsafe_allow_html=True
)

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data")

climate_files = {
    "Mbeya": "mbeya.xlsx",
    "Kongwa": "kongwa.xlsx",
    "Zanzibar": "zanzibar.xlsx"
}
yield_files = {
    "Mbeya": "maize_mbeya.xlsx",
    "Kongwa": "maize_kongwa.xlsx",
    "Zanzibar": "maize_zanzibar.xlsx"
}

# Daily files
rain_path = os.path.join(data_path, "Rainfall.xlsx")
rh_path   = os.path.join(data_path, "Relative Humidity.xlsx")
tmax_path = os.path.join(data_path, "Tmax.xlsx")
tmin_path = os.path.join(data_path, "Tmin.xlsx")
# =========================
# CACHE DATA
# =========================

@st.cache_data
def load_monthly_data(file):
    return load_block_data(file)


@st.cache_data
def load_yield_data(file):
    df = pd.read_excel(file)
    df.columns = (
        df.columns
        .str.upper()
        .str.strip()
    )
    return df
# =========================
# REGION
# =========================
region = st.selectbox(
    T("📍 Select Region", "📍 Chagua Mkoa"),
    list(climate_files.keys()),
    key="main_region_select"
)
# =========================
# TEST LIVE WEATHER
# =========================

try:

    with st.spinner(
        T(
            "🌤 Fetching live weather data...",
            "🌤 Inapakua taarifa za hali ya hewa..."
        )
    ):
        weather = get_weather(region)

except Exception:
    weather = None

if weather:

    st.success(
        T(
            "✅ Live Weather Connected",
            "✅ Taarifa za hali ya hewa zimepatikana"
        )
    )

    st.caption(
        T(
            f"Last updated: {datetime.now().strftime('%d %b %Y %H:%M')}",
            f"Imesasishwa: {datetime.now().strftime('%d %b %Y %H:%M')}"
        )
    )

    st.subheader(
        T(
            "🌤 Current Weather",
            "🌤 Hali ya Hewa ya Sasa"
        )
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("🌡 Temperature", f"{weather['temperature']:.1f} °C")
        st.metric("💧 Humidity", f"{weather['humidity']} %")

    with c2:
        st.metric("🌧 Rainfall", f"{weather['rainfall']} mm")
        st.metric("🌬 Wind", f"{weather['wind']} m/s")

    with c3:
        st.metric("📈 Pressure", f"{weather['pressure']} hPa")
        st.metric("🔥 Max Temp", f"{weather['temp_max']} °C")

else:

    st.error(
        T(
            "Weather API failed.",
            "Imeshindikana kupata taarifa za hali ya hewa."
        )
    )
monthly_file = os.path.join(
    data_path,
    climate_files[region]
)

yield_file = os.path.join(
    data_path,
    yield_files[region]
)



# =========================
# LOAD MONTHLY DATA
# =========================

df_month = load_monthly_data(monthly_file)
# =========================
# MONTHLY COMPUTATIONS
# =========================

df_month = compute_vpd(df_month)


df_month["FAW"] = df_month.apply(
    lambda r: faw_risk(
        r["TMEAN"],
        r["RH"],
        r["RAINFALL"]
    ),
    axis=1
)


df_month["VPD_STATE"] = df_month["VPD"].apply(
    lambda x:
        "Dry" if x > 1.2
        else "Good" if x > 0.5
        else "Humid"
)


df_month["RAIN_STATE"] = df_month["RAINFALL"].apply(
    rainfall_state
)
# =========================
# DAILY DATA FROM API
# =========================

if weather is not None:

    df_daily = pd.DataFrame({

        "RAINFALL": [weather["rainfall"]],
        "RH": [weather["humidity"]],
        "TMAX": [weather["temp_max"]],
        "TMIN": [weather["temp_min"]]

    })

else:

    st.error("Unable to retrieve weather data.")
    st.stop()

# =========================
# CLEAN DATA
# =========================

for col in ["RAINFALL", "RH", "TMAX", "TMIN"]:
    df_daily[col] = pd.to_numeric(
        df_daily[col],
        errors="coerce"
    )

df_daily = df_daily.dropna()

# =========================
# DAILY COMPUTATIONS
# =========================

df_daily = compute_vpd(df_daily)

df_daily["FAW"] = df_daily.apply(
    lambda r: faw_risk(
        r["TMEAN"],
        r["RH"],
        r["RAINFALL"]
    ),
    axis=1
)

df_daily["VPD_STATE"] = df_daily["VPD"].apply(
    lambda x:
        "Dry" if x > 1.2
        else "Good" if x > 0.5
        else "Humid"
)

df_daily["RAIN_STATE"] = df_daily["RAINFALL"].apply(
    rainfall_state
)

health = calculate_crop_health(
    weather,
    df_daily.iloc[0]["VPD"],
    df_daily.iloc[0]["FAW"]
)

# =========================
# CLIMATE ADJUSTMENT
# =========================

adjustment = climate_adjustment(
    weather,
    df_daily.iloc[0]["VPD"]
)

st.info(f"🌦️ Climate Adjustment Score: {adjustment}")

# =========================
# MONTHLY RAIN FORECAST
# =========================

monthly_rain_tr = build_transition_matrix(
    df_month["RAIN_STATE"]
)

# Current rain state
current_rain_m = df_month.iloc[-1]["RAIN_STATE"]

next_rain_m = predict_next(
    current_rain_m,
    monthly_rain_tr
)

# Climate corrected forecast
next_rain_m = adjust_monthly_rain(
    next_rain_m,
    adjustment
)
# =========================
# MONTHLY VPD FORECAST
# =========================

monthly_vpd_tr = build_transition_matrix(
    df_month["VPD_STATE"]
)

current_vpd_m = df_month.iloc[-1]["VPD_STATE"]

next_vpd_m = predict_next(
    current_vpd_m,
    monthly_vpd_tr
)

# =========================
# MONTHLY FAW FORECAST
# =========================

monthly_faw_tr = build_transition_matrix(
    df_month["FAW"]
)

current_faw_m = df_month.iloc[-1]["FAW"]

next_faw_m = predict_next(
    current_faw_m,
    monthly_faw_tr
)
# =========================
# SAVE TODAY'S WEATHER
# =========================

save_weather(
    weather,
    region,
    df_daily.iloc[0]["VPD"],
    df_daily.iloc[0]["FAW"]
)
# ======================================
# AI CROP HEALTH
# ======================================

st.subheader(

    T(
        "🧠 AI Crop Health Assessment",
        "🧠 Tathmini ya AI ya Afya ya Mazao"
    )

)

c1, c2 = st.columns(2)

with c1:

    st.metric(
        T(
            "🌧 Rainfall Status",
            "🌧 Hali ya Mvua"
        ),
        f"{health['rain']}%"
    )

    st.metric(
        T(
            "🌡 Temperature Status",
            "🌡 Hali ya Joto"
        ),
        f"{health['temperature']}%"
    )

with c2:

    st.metric(
        T(
            "💧 Water Status",
            "💧 Hali ya Maji"
        ),
        f"{health['water']}%"
    )

    st.metric(
        T(
            "🐛 FAW Status",
            "🐛 Hali ya FAW"
        ),
        f"{health['faw']}%"
    )
st.metric(

    T(
        "🌿 Overall Crop Health",
        "🌿 Afya ya Jumla ya Mazao"
    ),

    f"{health['overall']}%"

)

st.progress(health["overall"] / 100)
if health["overall"] >= 80:

    st.success(

        T(

            "Excellent crop conditions. Continue routine field monitoring.",

            "Hali ya mazao ni nzuri sana. Endelea kufuatilia shamba kama kawaida."

        )

    )

elif health["overall"] >= 60:

    st.info(

        T(

            "Moderate crop conditions. Continue monitoring and field scouting.",

            "Hali ya mazao ni ya wastani. Endelea kufuatilia shamba na kufanya ukaguzi wa mara kwa mara."

        )

    )

else:

    st.error(

        T(

            "Poor crop conditions detected. Immediate intervention is recommended.",

            "Hali ya mazao si nzuri. Hatua za haraka zinapendekezwa ili kupunguza athari."

        )

    )
# ======================================
# CROPPING SEASON BY REGION
# ======================================

from datetime import datetime

def get_cropping_season(region):

    month = datetime.now().month

    if region == "Mbeya":

        return {
            "name": "Main Season",
            "start_month": 11,
            "end_month": 4
        }

    elif region == "Kongwa":

        return {
            "name": "Main Season",
            "start_month": 12,
            "end_month": 5
        }

    elif region == "Zanzibar":

        # MASIKA (March-June)
        if month in [3, 4, 5, 6]:
            return {
                "name": "Masika",
                "start_month": 3,
                "end_month": 6
            }

        # VULI (October-January)
        else:
            return {
                "name": "Vuli",
                "start_month": 10,
                "end_month": 1
            }

    return {
        "name": "Main Season",
        "start_month": 11,
        "end_month": 4
    }


# ======================================
# CROP CALENDAR
# ======================================

def get_crop_stage(region):

    month = datetime.now().month

    calendars = {

        "Mbeya": {

            11: ("🌱 Planting", "🌱 Kupanda"),
            12: ("🌿 Vegetative", "🌿 Ukuaji"),
            1: ("🌿 Vegetative", "🌿 Ukuaji"),
            2: ("🌾 Flowering", "🌾 Maua"),
            3: ("🌽 Grain Filling", "🌽 Kujaza Punje"),
            4: ("🚜 Harvest", "🚜 Mavuno")

        },

        "Kongwa": {

            12: ("🌱 Planting", "🌱 Kupanda"),
            1: ("🌿 Vegetative", "🌿 Ukuaji"),
            2: ("🌿 Vegetative", "🌿 Ukuaji"),
            3: ("🌾 Flowering", "🌾 Maua"),
            4: ("🌽 Grain Filling", "🌽 Kujaza Punje"),
            5: ("🚜 Harvest", "🚜 Mavuno")

        },

        "Zanzibar": {

            # MASIKA
            3: ("🌱 Planting (Masika)", "🌱 Kupanda (Masika)"),
            4: ("🌿 Vegetative", "🌿 Ukuaji"),
            5: ("🌾 Flowering", "🌾 Maua"),
            6: ("🚜 Harvest", "🚜 Mavuno"),

            # VULI
            10: ("🌱 Planting (Vuli)", "🌱 Kupanda (Vuli)"),
            11: ("🌿 Vegetative", "🌿 Ukuaji"),
            12: ("🌾 Flowering", "🌾 Maua"),
            1: ("🚜 Harvest", "🚜 Mavuno")

        }

    }

    return calendars.get(region, {}).get(
        month,
        (
            "🌱 Off Season",
            "🌱 Nje ya Msimu"
        )
    )


# ======================================
# SEASON PROGRESS
# ======================================

def get_season_progress(region):

    month = datetime.now().month

    progress = {

        "Mbeya": {

            11: 15,
            12: 30,
            1: 45,
            2: 65,
            3: 85,
            4: 100

        },

        "Kongwa": {

            12: 15,
            1: 30,
            2: 50,
            3: 70,
            4: 90,
            5: 100

        },

        "Zanzibar": {

            # MASIKA
            3: 20,
            4: 50,
            5: 80,
            6: 100,

            # VULI
            10: 20,
            11: 50,
            12: 80,
            1: 100

        }

    }

    return progress.get(region, {}).get(month, 0)
# ======================================
# GET CURRENT SEASON DATA
# ======================================

def get_season_data(region):

    season = get_cropping_season(region)

    start = season["start_month"]
    end = season["end_month"]

    current_year = datetime.now().year
    current_month = datetime.now().month

    if start > end:

        if current_month >= start:

            start_date = f"{current_year}-{start:02d}-01"
            end_date = f"{current_year+1}-{end:02d}-31"

        else:

            start_date = f"{current_year-1}-{start:02d}-01"
            end_date = f"{current_year}-{end:02d}-31"

    else:

        start_date = f"{current_year}-{start:02d}-01"
        end_date = f"{current_year}-{end:02d}-31"

    query = """

    SELECT *

    FROM weather_history

    WHERE region=?

    AND date BETWEEN ? AND ?

    """

    season_df = pd.read_sql(
        query,
        conn,
        params=(region, start_date, end_date)
    )

    return season_df

# ======================================
# DETERMINE CURRENT SEASON
# ======================================

current_month = datetime.now().month

season = get_cropping_season(region)

start = season["start_month"]
end = season["end_month"]

if region == "Zanzibar":

    # Masika active
    if 3 <= current_month <= 6:

        season_name = "Masika"
        start = 3
        end = 6
        in_season = True

    # Vuli active
    elif current_month >= 10 or current_month == 1:

        season_name = "Vuli"
        start = 10
        end = 1
        in_season = True

    # Between Masika and Vuli
    elif 7 <= current_month <= 9:

        season_name = "Vuli"
        start = 10
        end = 1
        in_season = False

    # February
    else:

        season_name = "Masika"
        start = 3
        end = 6
        in_season = False

else:

    season_name = season["name"]

    if start > end:

        in_season = (
            current_month >= start or
            current_month <= end
        )

    else:

        in_season = (
            start <= current_month <= end
        )
# ======================================
# AI PLANTING ADVISORY
# ======================================

st.subheader(
    T(
        "🌱 AI Planting Advisory",
        "🌱 Ushauri wa AI kuhusu Upandaji"
    )
)

if not in_season:

    months_remaining = (start - current_month) % 12

    st.info(
        T(
            f"The {season_name} cropping season has not started yet.\n\n"
            f"🌧 Expected onset: Month {start}\n"
            f"⏳ Approximately {months_remaining} month(s) remaining.\n\n"
            "Recommendation: Prepare land, secure seeds and continue monitoring seasonal weather updates.",
            
            f"Msimu wa {season_name} bado haujaanza.\n\n"
            f"🌧 Mvua zinatarajiwa kuanza mwezi wa {start}.\n"
            f"⏳ Zimebaki takribani mwezi {months_remaining}.\n\n"
            "Ushauri: Endelea kuandaa shamba, pata mbegu na fuatilia taarifa za hali ya hewa."
        )
    )

else:

    st.success(
        T(
            "✅ {season_name} cropping season is active.\n\n"
            "Recommendation:\n"
            "• Begin planting if adequate rainfall has been received.\n"
            "• Continue monitoring daily weather updates.\n"
            "• Monitor Fall Armyworm risk.",
            
            "✅ Msimu wa {season_name} unaendelea.\n\n"
            "Ushauri:\n"
            "• Panda ikiwa mvua za kutosha zimepatikana.\n"
            "• Endelea kufuatilia taarifa za hali ya hewa kila siku.\n"
            "• Endelea kufuatilia hatari ya Fall Armyworm."
        )
    )
# Default values
season_rain = 0
avg_temp = 0
avg_vpd = 0
high_faw_days = 0
medium_faw_days = 0
days_recorded = 0

# ======================================
# DISPLAY SEASON INFORMATION
# ======================================

st.subheader(
    T(
        f"🌾 Season Tracker ({season_name})",
        f"🌾 Ufuatiliaji wa Msimu ({season_name})"
    )
)

if not in_season:

    st.info(

        T(

            "🌱 The cropping season is currently inactive in this region.",

            "🌱 Kwa sasa msimu wa kilimo haujaanza katika eneo hili."

        )

    )

elif season_df.empty:

    st.warning(

        T(

            "⚠ No seasonal weather records are available yet. Data will be collected automatically each day.",

            "⚠ Bado hakuna kumbukumbu za hali ya hewa za msimu. Mfumo utaendelea kuhifadhi taarifa za hali ya hewa kila siku."

        )

    )

else:

    season_rain = season_df["rainfall"].sum()

    avg_temp = season_df["temperature"].mean()

    avg_vpd = season_df["vpd"].mean()

    high_faw_days = (season_df["faw"] == "High").sum()

    medium_faw_days = (season_df["faw"] == "Medium").sum()

    days_recorded = len(season_df)
# ======================================
# CROP CALENDAR
# ======================================

st.subheader(

    T(

        "🌽 Crop Calendar",

        "🌽 Kalenda ya Kilimo"

    )

)

stage_en, stage_sw = get_crop_stage(region)

progress = get_season_progress(region)

c1, c2 = st.columns(2)

with c1:

    st.metric(

        T(

            "Current Stage",

            "Hatua ya Mazao"

        ),

        stage_en if language == "English" else stage_sw

    )

with c2:

    st.metric(

        T(

            "Season Progress",

            "Maendeleo ya Msimu"

        ),

        f"{progress}%"

    )

st.progress(progress / 100)

# ======================================
# SEASON DASHBOARD
# ======================================

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(

        T(

            "🌧 Season Rainfall",

            "🌧 Mvua ya Msimu"

        ),

        f"{season_rain:.1f} mm"

    )

    st.metric(

        T(

            "📅 Days Recorded",

            "📅 Siku Zilizorekodiwa"

        ),

        days_recorded

    )

with c2:

    st.metric(

        T(

            "🌡 Average Temperature",

            "🌡 Wastani wa Joto"

        ),

        f"{avg_temp:.1f} °C"

    )

    st.metric(

        T(

            "💧 Average VPD",

            "💧 Wastani wa VPD"

        ),

        f"{avg_vpd:.2f}"

    )

with c3:

    st.metric(

        T(

            "🐛 High FAW Days",

            "🐛 Siku zenye Hatari Kubwa ya FAW"

        ),

        high_faw_days

    )

    st.metric(

        T(

            "⚠ Medium FAW Days",

            "⚠ Siku zenye Hatari ya Kati ya FAW"

        ),

        medium_faw_days

    )
# =========================
# DAILY MARKOV
# =========================
daily_vpd_tr = build_transition_matrix(
    df_daily["VPD_STATE"]
)

daily_faw_tr = build_transition_matrix(
    df_daily["FAW"]
)

current_vpd_d = df_daily.iloc[-1]["VPD_STATE"]
current_faw_d = df_daily.iloc[-1]["FAW"]

next_vpd_d = predict_next(
    current_vpd_d,
    daily_vpd_tr
)

next_faw_d = predict_next(
    current_faw_d,
    daily_faw_tr
)
daily_rain_tr = build_transition_matrix(
    df_daily["RAIN_STATE"]
)

current_rain_d = df_daily.iloc[-1]["RAIN_STATE"]

next_rain_d = predict_next(
    current_rain_d,
    daily_rain_tr
)
# =========================
# DAILY FORECAST AI CARDS
# =========================
st.subheader(T(
    "🌦️ Daily Forecast",
    "🌦️ Utabiri wa Kila Siku"
))

col1, col2, col3 = st.columns(
    [1,1,1],
    gap="large"
)

# =========================
# FAW CARD
# =========================
with col1:

    faw_title = (
        "🌽 Hatari ya Viwavijeshi Vamizi"
        if language == "Kiswahili"
        else "🌽 FAW Risk"
    )

    st.markdown(
        f"""
        <div class="card">
        <div class="title">
        {faw_title}
        </div>

        <div class="state">
        {translate_faw(next_faw_d, language)}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        ai_image("FAW", next_faw_d),
        use_container_width=True
    )

# =========================
# VPD CARD
# =========================
with col2:

    vpd_title = (
        "🌡️ Hali ya Unyevu"
        if language == "Kiswahili"
        else "🌡️ VPD State"
    )

    st.markdown(
        f"""
        <div class="card">
        <div class="title">
        {vpd_title}
        </div>

        <div class="state">
        {translate_vpd(next_vpd_d, language)}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        ai_image("VPD", next_vpd_d),
        use_container_width=True
    )

# =========================
# RAIN CARD
# =========================
with col3:

    rain_title = (
        "🌧️ Utabiri wa Mvua"
        if language == "Kiswahili"
        else "🌧️ Rain Forecast"
    )

    st.markdown(
        f"""
        <div class="card">
        <div class="title">
        {rain_title}
        </div>

        <div class="state">
        {translate_rain(next_rain_d, language)}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        ai_image("RAIN", next_rain_d),
        use_container_width=True
    )
# =========================
# MONTHLY FORECAST AI CARDS
# =========================
st.subheader(T(
    "📊 Monthly Forecast",
    "📊 Utabiri wa Mwezi"
))

col1, col2, col3 = st.columns(3)

# =========================
# FAW MONTHLY CARD
# =========================
with col1:

    monthly_faw_title = (
        "🌽 Hatari ya Viwavijeshi Vamizi kwa Mwezi"
        if language == "Kiswahili"
        else "🌽 Monthly FAW Risk"
    )

    st.markdown(
        f"""
        <div class="card">
        <div class="title">
        {monthly_faw_title}
        </div>

        <div class="state">
        {translate_faw(next_faw_m, language)}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        ai_image("FAW", next_faw_m),
        use_container_width=True
    )

# =========================
# VPD MONTHLY CARD
# =========================
with col2:

    monthly_vpd_title = (
        "🌡️ Hali ya Unyevu kwa Mwezi"
        if language == "Kiswahili"
        else "🌡️ Monthly VPD State"
    )

    st.markdown(
        f"""
        <div class="card">
        <div class="title">
        {monthly_vpd_title}
        </div>

        <div class="state">
        {translate_vpd(next_vpd_m, language)}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        ai_image("VPD", next_vpd_m),
        use_container_width=True
    )

# =========================
# RAIN MONTHLY CARD
# =========================
with col3:

    monthly_rain_title = (
        "🌧️ Utabiri wa Mvua kwa Mwezi"
        if language == "Kiswahili"
        else "🌧️ Monthly Rain Outlook"
    )

    st.markdown(
        f"""
        <div class="card">
        <div class="title">
        {monthly_rain_title}
        </div>

        <div class="state">
        {translate_rain(next_rain_m, language)}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        ai_image("RAIN", next_rain_m),
        use_container_width=True
    )
# =========================
# YIELD PREDICTION
# =========================

st.subheader(T(
    "🌽 Yield Prediction",
    "🌽 Utabiri wa Mavuno"
))


if os.path.exists(yield_file):

    yield_df = load_yield_data(yield_file)
# =========================
# YEARLY CLIMATE FEATURES
# =========================

model_df = prepare_annual_features(
    df_month,
    yield_df
)


# =========================
# TRAIN YIELD MODEL
# =========================

model_df = (
    model_df
    .dropna()
    .reset_index(drop=True)
)

features = get_region_features(region)

if region == "Mbeya":

    X, y = prepare_training_data(
        model_df,
        features
    )

    models = get_models()

    if len(model_df) > 8:

        (
            best_model,
            best_name,
            best_r2,
            best_rmse,
            best_mae
        ) = train_models(
            models,
            X,
            y
        )
# =========================
# TRAIN FINAL MODEL
# =========================

if region == "Mbeya":

    tmean = (weather["temp_max"] + weather["temp_min"]) / 2

    faw_score = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }[df_daily.iloc[0]["FAW"]]

    latest_data = pd.DataFrame({
        "TMEAN": [tmean],
        "RAINFALL_SUM": [weather["rainfall"]],
        "SEASON_RAIN": [model_df.iloc[-1]["SEASON_RAIN"]],
        "SEASONAL_RAIN": [model_df.iloc[-1]["SEASONAL_RAIN"]],
        "RAIN_ANOMALY": [
            weather["rainfall"] - model_df["RAINFALL_SUM"].mean()
        ],
        "VPD": [df_daily.iloc[0]["VPD"]],
        "FAW_SCORE": [faw_score],
        "YIELD_LAG1": [model_df.iloc[-1]["YIELD"]],
        "RAIN_LAG1": [model_df.iloc[-1]["RAINFALL_SUM"]],
        "VPD_LAG1": [model_df.iloc[-1]["VPD"]]
    })

    latest_data = latest_data[features]

    prediction = predict_yield(
        best_model,
        X,
        y,
        latest_data
    )

    historical_average = yield_df["YIELD"].mean()

    # hapa inaendelea code yako yote ya
    # Yield Forecast
    # st.metric(...)
    # st.success(...)
    # st.warning(...)
else:

    st.subheader(
        T(
            "🌱 AI Climate Advisory",
            "🌱 Ushauri wa AI wa Hali ya Hewa"
        )
    )

    if health["overall"] >= 80:

        st.success(
            T(
                "✅ Weather conditions are favourable. Planting can begin after effective rainfall.",
                "✅ Hali ya hewa ni nzuri. Panda baada ya mvua za kutosha kuanza."
            )
        )

    elif health["overall"] >= 60:

        st.info(
            T(
                "ℹ Moderate weather conditions. Continue monitoring rainfall before planting.",
                "ℹ Hali ya hewa ni ya wastani. Endelea kufuatilia mvua kabla ya kupanda."
            )
        )

    else:

        st.warning(
            T(
                "⚠ Weather conditions are currently not favourable. Delay planting and continue monitoring.",
                "⚠ Hali ya hewa bado si nzuri kwa kupanda. Endelea kufuatilia taarifa za hali ya hewa."
            )
        )

# =========================
# AI SMART RECOMMENDATIONS
# =========================
st.subheader(T(
    "🤖 Recommendations",
    "🤖 Ushauri"
))

# FAW Recommendations
if current_faw_d == "High":

    st.error(T(
        "⚠️ High Fall Armyworm risk detected.",
        "⚠️ Hatari kubwa ya ViwaviJeshi Vamizi imegunduliwa."
    ))

    st.write(T(
        """
        • Inspect maize fields immediately  
        • Apply recommended pesticides or biocontrol methods  
        • Remove heavily infected plants  
        • Increase field monitoring frequency  
        • Inform nearby farmers\n
        • Consult an Agricultural Extension Officer for further advice and assistance
        """,
        """
        • Kagua shamba haraka iwezekanavyo.  
        • Tumia viuatilifu/dawa vinavyopendekezwa  
        • Ondoa mimea iliyoathirika sana  
        • Ongeza ufuatiliaji wa shamba  
        • Wajulishe wakulima wa karibu\n
        • Muone Afisa Ugani/bibi au bwana shamba kwa ushauri na msaada zaidi
        """
    ))

elif current_faw_d == "Medium":

    st.warning(T(
        "⚠️ Moderate Fall Armyworm risk detected.",
        "⚠️ Hatari ya wastani ya ViwaviJeshi Vamizi imegunduliwa."
    ))

    st.write(T(
        """
        • Continue monitoring crop conditions  
        • Use early preventive control measures  
        • Check leaves for larvae and egg masses  
        • Maintain proper field sanitation\n
        • Consult an Agricultural Extension Officer for further advice and assistance
        """,
        """
        • Endelea kufuatilia hali ya mazao  
        • Tumia hatua za mapema za kinga  
        • Kagua majani kwa mabuu(mafunza) na mayai  
        • Dumisha usafi wa shamba\n
        • Muone Afisa Ugani/bibi au bwana shamba kwa ushauri na msaada zaidi

        """
    ))

else:

    st.success(T(
        "✅ Low Fall Armyworm risk.",
        "✅ Hatari ndogo ya ViwaviJeshi Vamizi."
    ))

    st.write(T(
        """
        • Maintain good agricultural practices (GAP)  
        • Continue routine monitoring  
        """,
        """
        • Endelea kufuata mbinu bora za kilimo.  
        • Fanya ufuatiliaji wa kawaida.

        """
    ))

# =========================
# VPD RECOMMENDATIONS
# =========================
if current_vpd_d == "Dry":

    st.error(T(
        "🌡️ High dryness stress detected.",
        "🌡️ Ukame mkubwa umeonekana."
    ))

    st.write(T(
        """
        • Increase irrigation if available  
        • Apply mulching to conserve moisture  
        • Monitor signs of wilting
        """,
        """
        • Ongeza umwagiliaji kama inawezekana  
        • Tumia matandazo kuhifadhi unyevu    
        • Fuatilia dalili za kunyauka
        """
    ))

elif current_vpd_d == "Humid":

    st.warning(T(
        "💧 Excess humidity detected.",
        "💧 Unyevunyevu mkubwa umeonekana."
    ))

    st.write(T(
        """
        • Improve field ventilation (Maintain reccomended spacing) 
        • Monitor fungal disease outbreaks  
        • Avoid excessive irrigation  
        • Ensure proper drainage
        """,
        """
        • Boresha mzunguko wa hewa shambani (Panda kwa nafasi zilizopendekezwa)  
        • Fuatilia magonjwa ya fangasi/Ukungu  
        • Epuka umwagiliaji kupita kiasi  
        • Hakikisha unaondoa maji yaliyotuama shambani
        """
    ))

else:

    st.success(T(
        "✅ Favorable VPD conditions detected.",
        "✅ Hali nzuri ya Unyevu imeonekana."
    ))

    st.write(T(
        """
        • Continue standard crop management  
        • Maintain balanced irrigation  
        • Monitor weather changes regularly
        """,
        """
        • Endelea na usimamizi wa kawaida wa mazao  
        • Dumisha umwagiliaji wa wastani  
        • Fuatilia mabadiliko ya hali ya hewa mara kwa mara.
        """
    ))
# =========================
# FOOTER
# =========================
st.write("---")

st.caption(T(
    "AI Early Warning System – Tanzania 🌽",
    "Mfumo wa AI wa Tahadhari ya Mapema – Tanzania 🌽"
))