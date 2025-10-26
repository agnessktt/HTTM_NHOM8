import os
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# -----------------------------
# Khởi tạo FastAPI
app = FastAPI(title="AirCare AQI + Health Alert API")

# -----------------------------
# Load model RandomForest
MODEL_PATH = os.path.abspath("models/aqi_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Không tìm thấy model tại {MODEL_PATH}. Hãy chạy train_model.py trước."
    )

model = joblib.load(MODEL_PATH)
print(f"💾 Model đã được load từ: {MODEL_PATH}")

# -----------------------------
# Khởi tạo OpenAI client
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")  # Thay bằng API key thật

# -----------------------------
# Hàm sinh cảnh báo sức khỏe cá nhân hóa
def generate_health_alert(aqi: float, user_info: dict = None) -> str:
    """
    Sinh cảnh báo sức khỏe dựa trên AQI và thông tin người dùng.
    """
    prompt = f"AQI hiện tại là {aqi}. "
    if user_info:
        prompt += f"Người dùng: {user_info}. "
    prompt += (
        "Viết thông báo ngắn gọn bằng tiếng Việt, tối đa 3 câu, "
        "cảnh báo mức độ nguy hiểm và khuyến nghị hành động phù hợp."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    alert_text = response.choices[0].message.content
    return alert_text

# Input schema
class AQIInput(BaseModel):
    # Dữ liệu môi trường
    pm2_5: float
    pm10: float
    no2: float
    co: float
    o3: float
    so2: float
    temp: float
    humidity: float
    # Thông tin người dùng (tùy chọn)
    age: int = None
    disease: str = None

# Endpoint POST /predict_with_alert
@app.post("/predict_with_alert")
def predict_with_alert(payload: AQIInput):
    # Tạo DataFrame từ input
    df = pd.DataFrame([{
        "pm2_5": payload.pm2_5,
        "pm10": payload.pm10,
        "no2": payload.no2,
        "co": payload.co,
        "o3": payload.o3,
        "so2": payload.so2,
        "temp": payload.temp,
        "humidity": payload.humidity
    }])

    # Dự đoán AQI
    aqi_pred = model.predict(df)[0]

    # Chuẩn bị thông tin người dùng
    user_info = {}
    if payload.age is not None:
        user_info["age"] = payload.age
    if payload.disease:
        user_info["disease"] = payload.disease

    # Sinh cảnh báo sức khỏe cá nhân hóa
    alert = generate_health_alert(aqi_pred, user_info)

    return {
        "aqi_pred": round(aqi_pred, 2),
        "health_alert": alert
    }

# Endpoint GET / để test server
@app.get("/")
def root():
    return {"message": "AirCare AQI + Health Alert API đang chạy!"}
