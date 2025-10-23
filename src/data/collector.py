import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import time

# ---------------------------
# 🔧 Cấu hình
# ---------------------------
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = os.getenv("DEFAULT_CITY", "Hanoi")
RAW_PATH = "data/raw/air_data.csv"
INTERVAL = 120  # Lấy dữ liệu mỗi 2 phút (120 giây)

# ---------------------------
# 🔹 Hàm lấy dữ liệu chất lượng không khí và thời tiết
# ---------------------------
def fetch_air_quality(city=CITY):
    print(f"📡 Đang lấy dữ liệu không khí cho: {city}...")

    # Lấy tọa độ
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    geo_resp = requests.get(geo_url).json()

    if not isinstance(geo_resp, list) or len(geo_resp) == 0:
        raise ValueError(f"❌ Không tìm thấy thành phố '{city}'.")

    lat, lon = geo_resp[0]["lat"], geo_resp[0]["lon"]

    # Dữ liệu không khí
    air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    air_data = requests.get(air_url).json()

    # Dữ liệu thời tiết
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    weather_data = requests.get(weather_url).json()

    # Xử lý dữ liệu
    components = air_data["list"][0]["components"]
    aqi = air_data["list"][0]["main"]["aqi"]
    temp = weather_data["main"]["temp"]
    humidity = weather_data["main"]["humidity"]

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "no2": components.get("no2"),
        "co": components.get("co"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "temp": temp,
        "humidity": humidity,
        "aqi": aqi * 50  # đổi thang 1–5 sang 50–250
    }

    # Lưu vào CSV
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df = pd.DataFrame([data])
    if not os.path.exists(RAW_PATH):
        df.to_csv(RAW_PATH, index=False)
    else:
        df.to_csv(RAW_PATH, mode="a", header=False, index=False)

    print(f"✅ Đã lưu dữ liệu vào {RAW_PATH} lúc {data['timestamp']}")
    return data

# ---------------------------
# 🔹 Vòng lặp tự động thu thập dữ liệu
# ---------------------------
if __name__ == "__main__":
    print("🚀 Bắt đầu thu thập dữ liệu không khí liên tục...")
    while True:
        try:
            fetch_air_quality()
        except Exception as e:
            print(f"❌ Lỗi: {e}")
        time.sleep(INTERVAL)
