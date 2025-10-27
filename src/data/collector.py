import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import time
import logging

# ---------------------------
# ⚙️ Cấu hình
# ---------------------------
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = os.getenv("DEFAULT_CITY", "Hanoi")
RAW_PATH = "data/raw/air_data.csv"
LOG_PATH = "data/logs/collector.log"
INTERVAL = int(os.getenv("FETCH_INTERVAL", 120))  # thời gian lấy dữ liệu
MAX_RUNS = int(os.getenv("MAX_RUNS", 0))  # 0 = chạy mãi, >0 = số lần lấy dữ liệu rồi dừng

# ---------------------------
# 🧾 Cấu hình logging
# ---------------------------
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------
# 🔹 Hàm lấy dữ liệu không khí & thời tiết
# ---------------------------
def fetch_air_quality(city=CITY):
    print("----------------------------------------------------")
    print(f"📡 [{datetime.now().strftime('%H:%M:%S')}] Lấy dữ liệu cho: {city}")

    try:
        # Lấy tọa độ
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        geo_resp = requests.get(geo_url, timeout=10).json()
        if not geo_resp:
            raise ValueError(f"Không tìm thấy thành phố '{city}'.")

        lat, lon = geo_resp[0]["lat"], geo_resp[0]["lon"]

        # Lấy dữ liệu không khí và thời tiết
        air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

        air_data = requests.get(air_url, timeout=10).json()
        weather_data = requests.get(weather_url, timeout=10).json()

        components = air_data["list"][0]["components"]
        aqi = air_data["list"][0]["main"]["aqi"]
        temp = weather_data["main"]["temp"]
        humidity = weather_data["main"]["humidity"]

        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city": city,
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "no2": components.get("no2"),
            "co": components.get("co"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "temp": temp,
            "humidity": humidity,
            "aqi": aqi * 50
        }

        # Lưu vào CSV (chống trùng timestamp)
        os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
        if os.path.exists(RAW_PATH):
            old_df = pd.read_csv(RAW_PATH)
            if data["timestamp"] in old_df["timestamp"].values:
                print("⚠️  Dữ liệu trùng thời gian, bỏ qua lần ghi này.")
                return data
            df = pd.concat([old_df, pd.DataFrame([data])], ignore_index=True)
        else:
            df = pd.DataFrame([data])
        df.to_csv(RAW_PATH, index=False)

        print(f"✅ Đã lưu vào {RAW_PATH} lúc {data['timestamp']}")
        print(f"🌡️  {temp}°C | 💧 {humidity}% | AQI: {data['aqi']}")
        logging.info(f"Saved new data for {city}: AQI={data['aqi']}, Temp={temp}, Humidity={humidity}")
        print("----------------------------------------------------\n")
        return data

    except Exception as e:
        logging.error(f"Lỗi khi lấy dữ liệu: {e}")
        print(f"❌ Lỗi: {e}")
        print("----------------------------------------------------\n")
        return None


# ---------------------------
# 🔁 Vòng lặp tự động thu thập
# ---------------------------
if __name__ == "__main__":
    print("🚀 Bắt đầu thu thập dữ liệu không khí liên tục...")
    print(f"🌆 Thành phố: {CITY}")
    print(f"⏳ Mỗi {INTERVAL} giây sẽ lấy và lưu dữ liệu mới.\n")

    count = 0
    while True:
        fetch_air_quality()
        count += 1
        if MAX_RUNS > 0 and count >= MAX_RUNS:
            print(f"✅ Đã chạy đủ {MAX_RUNS} lần, dừng chương trình.")
            break
        time.sleep(INTERVAL)
