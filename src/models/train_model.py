import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data/raw/air_data.csv"
MODEL_PATH = "models/aqi_model.pkl"

def train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại {DATA_PATH}")

    print(f"📖 Đang đọc dữ liệu từ {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    print("📊 Dữ liệu hiện tại:")
    print(df.tail())

    # Các cột đặc trưng đầu vào (features)
    features = ["pm2_5", "pm10", "no2", "co", "o3", "so2", "temp", "humidity"]
    target = "aqi"

    X = df[features]
    y = df[target]

    # Tách dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình
    print("🚀 Đang huấn luyện mô hình RandomForest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Dự đoán và đánh giá
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ MAE: {mae:.2f}")
    print(f"✅ R²: {r2:.2f}")

    # Lưu mô hình
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Mô hình đã được lưu tại: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
