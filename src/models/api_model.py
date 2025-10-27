import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from datetime import datetime

# -----------------------------
# Khởi tạo FastAPI
app = FastAPI(title="AirCare AQI + Health Alert API")

# -----------------------------
# Load model
MODEL_PATH = os.path.abspath("models/aqi_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Không tìm thấy model tại {MODEL_PATH}. Hãy chạy train_model.py trước."
    )

try:
    model = joblib.load(MODEL_PATH)
    print(f"💾 Model đã được load từ: {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Lỗi khi load model: {e}")

# -----------------------------
# Khởi tạo OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"))

# -----------------------------
# Xác định mức độ AQI
def get_aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Tốt"
    elif aqi <= 100:
        return "Trung bình"
    elif aqi <= 150:
        return "Kém"
    elif aqi <= 200:
        return "Xấu"
    else:
        return "Rất nguy hiểm"

# -----------------------------
# Sinh cảnh báo sức khỏe cá nhân hóa
def generate_health_alert(aqi: float, user_info: dict = None) -> str:
    category = get_aqi_category(aqi)
    base_prompt = (
        f"Chỉ số AQI hiện tại là {aqi:.1f} ({category}). "
        "Viết cảnh báo sức khỏe ngắn gọn bằng tiếng Việt, tối đa 3 câu, "
        "bao gồm mức độ nguy hiểm và khuyến nghị hành động phù hợp. "
    )

    if user_info:
        base_prompt += f"Thông tin người dùng: {user_info}. "

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": base_prompt}],
            temperature=0.7
        )
        alert_text = response.choices[0].message.content
        return alert_text
    except Exception as e:
        print(f"[{datetime.now()}] ⚠️ Lỗi OpenAI API: {e}")
        # Fallback nếu OpenAI lỗi
        return f"AQI hiện tại là {aqi:.1f} ({category}). Hạn chế ra ngoài và đeo khẩu trang nếu cần thiết."

# -----------------------------
# Input schema
class AQIInput(BaseModel):
    pm2_5: float
    pm10: float
    no2: float
    co: float
    o3: float
    so2: float
    temp: float
    humidity: float
    age: int = None
    disease: str = None

# -----------------------------
# Endpoint chính
@app.post("/predict_with_alert")
def predict_with_alert(payload: AQIInput):
    try:
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
        aqi_pred = float(model.predict(df)[0])

        # Chuẩn bị thông tin người dùng
        user_info = {}
        if payload.age is not None:
            user_info["tuổi"] = payload.age
        if payload.disease:
            user_info["bệnh lý"] = payload.disease

        # Sinh cảnh báo sức khỏe
        alert = generate_health_alert(aqi_pred, user_info)

        # Log request (tùy chọn)
        print(f"[{datetime.now()}] ✅ AQI={aqi_pred:.2f}, user={user_info}")

        return {
            "status": "success",
            "data": {
                "aqi_pred": round(aqi_pred, 2),
                "aqi_level": get_aqi_category(aqi_pred),
                "health_alert": alert
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

# -----------------------------
# Endpoint test server
@app.get("/")
def root():
    return {"message": "🌤 AirCare AQI + Health Alert API đang chạy tốt!"}
