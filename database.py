import sqlite3
import os
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "weather_history.db")

conn = sqlite3.connect(
    db_path,
    check_same_thread=False
)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS weather_history (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    date TEXT,

    region TEXT,

    temperature REAL,

    temp_max REAL,

    temp_min REAL,

    humidity REAL,

    rainfall REAL,

    pressure REAL,

    wind REAL,

    vpd REAL,

    faw TEXT

)
""")
def save_weather(weather, region, vpd, faw):

    today = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("""

    SELECT COUNT(*)

    FROM weather_history

    WHERE date=? AND region=?

    """, (today, region))

    exists = cursor.fetchone()[0]

    if exists == 0:

        cursor.execute("""

        INSERT INTO weather_history(

            date,
            region,
            temperature,
            temp_max,
            temp_min,
            humidity,
            rainfall,
            pressure,
            wind,
            vpd,
            faw

        )

        VALUES(?,?,?,?,?,?,?,?,?,?,?)

        """, (

            today,

            region,

            weather["temperature"],

            weather["temp_max"],

            weather["temp_min"],

            weather["humidity"],

            weather["rainfall"],

            weather["pressure"],

            weather["wind"],

            float(vpd),

            faw

        ))

        conn.commit()