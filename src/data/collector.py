import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import time
import logging

# ==========================================================
# ⚙️ CẤU HÌNH
# ==========================================================
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = os.getenv("DEFAULT_CITY", "Hanoi")
RAW_PATH = "data/raw/air_data.csv"
LOG_PATH = "data/logs/collector.log"
INTERVAL = 120  # Lấy dữ liệu mỗi 2 phút
TIMEZONE = timezone(timedelta(hours=7))  # Giờ Việt Nam (UTC+7)

# ==========================================================
# 🪵 CẤU HÌNH GHI LOG
# ==========================================================
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ==========================================================
# 🌫️ HÀM LẤY DỮ LIỆU KHÔNG KHÍ & THỜI TIẾT
# ==========================================================
def fetch_air_quality(city=CITY):
    logging.info(f"📡 Fetching air quality data for: {city}...")

    try:
        # --------- Lấy tọa độ ----------
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        geo_resp = requests.get(geo_url, timeout=10).json()

        if not isinstance(geo_resp, list) or len(geo_resp) == 0:
            raise ValueError(f"City '{city}' not found.")

        lat, lon = geo_resp[0]["lat"], geo_resp[0]["lon"]

        # --------- Lấy dữ liệu không khí ----------
        air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        air_data = requests.get(air_url, timeout=10).json()

        # --------- Lấy dữ liệu thời tiết ----------
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        weather_data = requests.get(weather_url, timeout=10).json()

        # --------- Xử lý dữ liệu ----------
        components = air_data["list"][0]["components"]
        aqi = air_data["list"][0]["main"]["aqi"]
        temp = weather_data["main"]["temp"]
        humidity = weather_data["main"]["humidity"]

        timestamp = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S%z")

        data = {
            "timestamp": timestamp,
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "no2": components.get("no2"),
            "co": components.get("co"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "temp": temp,
            "humidity": humidity,
            "aqi": aqi,
        }

        # --------- Lưu CSV ----------
        os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
        df = pd.DataFrame([data])

        if not os.path.exists(RAW_PATH):
            df.to_csv(RAW_PATH, index=False)
            logging.info("🆕 Created new data file: air_data.csv")
        else:
            df.to_csv(RAW_PATH, mode="a", header=False, index=False)

        logging.info(f"✅ Saved data at {timestamp} → PM2.5={data['pm2_5']:.2f}, AQI={aqi}")
        return data

    except Exception as e:
        logging.error(f"❌ Error while fetching data: {e}")
        raise

# ==========================================================
# 🔁 VÒNG LẶP THU THẬP TỰ ĐỘNG
# ==========================================================
if __name__ == "__main__":
    logging.info("🚀 Collector started — continuous air quality monitoring initialized.")
    print("🚀 Bắt đầu thu thập dữ liệu không khí liên tục...")
    
    while True:
        try:
            fetch_air_quality()
        except Exception as e:
            print(f"❌ Lỗi khi lấy dữ liệu: {e}")
        time.sleep(INTERVAL)
