import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Đường dẫn dữ liệu và lưu model
DATA_PATH = "data/raw/air_data.csv"
MODEL_PATH = "models/aqi_model.pkl"

def train_model():
    # Kiểm tra file dữ liệu
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại {DATA_PATH}")

    # ---- Sơ đồ luồng ----
    print("🟢 Bắt đầu luồng huấn luyện mô hình RandomForest")
    print("┌───────────────┐")
    print("│  Dữ liệu CSV  │")
    print("└───────┬───────┘")
    print("        ↓")
    print("┌───────────────┐")
    print("│  Train Model  │")
    print("└───────┬───────┘")
    print("        ↓")
    print("┌───────────────┐")
    print("│  Lưu Model    │")
    print("└───────┬───────┘")
    print("        ↓")
    print("┌───────────────┐")
    print("│  Dự đoán AQI  │")
    print("└───────────────┘\n")

    # 📖 Đọc dữ liệu
    print(f"📖 Đang đọc dữ liệu từ {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    print("📊 Dữ liệu hiện tại:")
    print(df.tail())

    # Các cột đặc trưng và target
    features = ["pm2_5", "pm10", "no2", "co", "o3", "so2", "temp", "humidity"]
    target = "aqi"

    X = df[features]
    y = df[target]

    # Tách dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🚀 Huấn luyện mô hình
    print("🚀 Đang huấn luyện mô hình RandomForest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Dự đoán và đánh giá
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # ✅ In kết quả
    print(f"✅ MAE: {mae:.2f}")
    print(f"✅ R²: {r2:.2f}")

    # 💾 Lưu model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Mô hình đã được lưu tại: {MODEL_PATH}\n")

    # ---- Dự đoán thử 1 mẫu ----
    sample_input = {"pm2_5": 35, "pm10": 50, "no2": 20, "co": 0.7,
                    "o3": 10, "so2": 5, "temp": 28, "humidity": 65}
    sample_df = pd.DataFrame([sample_input])
    sample_pred = model.predict(sample_df)
    print(f"📊 Dự đoán AQI cho mẫu thử: {sample_pred[0]:.2f}")

    return model, mae, r2

# Khi chạy trực tiếp
if __name__ == "__main__":
    train_model()
