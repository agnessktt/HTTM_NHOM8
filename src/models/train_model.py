import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ============================================================
# ⚙️ CẤU HÌNH
# ============================================================
DATA_PATH = "data/raw/air_data.csv"
MODEL_DIR = "models"
HORIZONS = [1, 3, 6]  # Dự báo 1h, 3h, 6h tới

# ============================================================
# 📥 1. NẠP DỮ LIỆU
# ============================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại {DATA_PATH}")

data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
data = data.drop_duplicates(subset="timestamp").sort_values("timestamp")
data = data.dropna(subset=["aqi"])

print(f"✅ Nạp dữ liệu: {len(data)} dòng, {data['timestamp'].min()} → {data['timestamp'].max()}")

# ============================================================
# 🧩 2. FEATURE ENGINEERING
# ============================================================
for col in ["pm2_5", "pm10", "co", "no2", "o3", "so2", "aqi"]:
    for lag in range(1, 4):  # tạo độ trễ 1–3 giờ
        data[f"{col}_lag{lag}"] = data[col].shift(lag)

data["aqi_rolling3"] = data["aqi"].rolling(window=3).mean()
data["hour"] = data["timestamp"].dt.hour
data["weekday"] = data["timestamp"].dt.weekday
data["month"] = data["timestamp"].dt.month
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

# ============================================================
# 🧠 3. HÀM HUẤN LUYỆN CHUNG
# ============================================================
def train_for_horizon(horizon: int):
    df = data.copy()
    df["aqi_future"] = df["aqi"].shift(-horizon)
    df = df.dropna()

    features = [c for c in df.columns if c not in ["timestamp", "aqi_future"]]
    X = df[features]
    y = df["aqi_future"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
        "verbose": -1,
    }

    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_test, y_test, reference=train_data)

    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, f"aqi_model_{horizon}h.pkl")
    joblib.dump(model, model_path)
    joblib.dump(features, os.path.join(MODEL_DIR, f"feature_names_{horizon}h.pkl"))

    print(f"✅ AQI +{horizon}h → MAE: {mae:.3f} | R²: {r2:.3f} | 📁 {os.path.basename(model_path)}")
    return {"horizon": horizon, "mae": mae, "r2": r2}

# ============================================================
# 🚀 4. HUẤN LUYỆN CẢ 3 MÔ HÌNH
# ============================================================
print("\n🚀 Bắt đầu huấn luyện mô hình dự báo AQI (1h, 3h, 6h)...\n")
results = [train_for_horizon(h) for h in HORIZONS]

# ============================================================
# 📊 5. TỔNG KẾT
# ============================================================
print("\n" + "=" * 60)
print("📈 HIỆU SUẤT CÁC MÔ HÌNH DỰ BÁO AQI")
print("=" * 60)
for r in results:
    print(f"⏱ {r['horizon']}h → MAE: {r['mae']:.3f} | R²: {r['r2']:.3f}")
print("=" * 60)
print("✅ Huấn luyện hoàn tất.\n")
