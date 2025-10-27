import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import logging
from dotenv import load_dotenv

# ============================================================
#  ⚙️ CẤU HÌNH
# ============================================================
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/raw/air_data.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
LOG_PATH = os.getenv("LOG_PATH", "data/logs/train.log")
HORIZONS = [1, 3, 6]

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
#  🧾 KIỂM TRA DỮ LIỆU
# ============================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại {DATA_PATH}")

data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
required_cols = {"timestamp", "aqi", "pm2_5", "pm10", "co", "no2", "o3", "so2", "temp", "humidity"}
if not required_cols.issubset(data.columns):
    raise ValueError(f"⚠️ Dữ liệu thiếu cột cần thiết! Cần có: {required_cols}")

data = data.drop_duplicates(subset="timestamp").sort_values("timestamp")
data = data.dropna(subset=["aqi"])

if len(data) < 100:
    logging.warning(f"Dữ liệu quá ít ({len(data)} dòng). Mô hình có thể không ổn định.")

print(f"✅ Nạp dữ liệu: {len(data)} dòng, {data['timestamp'].min()} → {data['timestamp'].max()}")

# ============================================================
#  🧠 FEATURE ENGINEERING
# ============================================================
for col in ["pm2_5", "pm10", "co", "no2", "o3", "so2", "aqi"]:
    for lag in range(1, 4):
        data[f"{col}_lag{lag}"] = data[col].shift(lag)

data["aqi_rolling3"] = data["aqi"].rolling(window=3).mean()
data["hour"] = data["timestamp"].dt.hour
data["weekday"] = data["timestamp"].dt.weekday
data["month"] = data["timestamp"].dt.month
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
data = data.dropna()

# ============================================================
#  🚀 HÀM HUẤN LUYỆN MỘT MÔ HÌNH
# ============================================================
def train_for_horizon(horizon: int):
    df = data.copy()
    df["aqi_future"] = df["aqi"].shift(-horizon)
    df = df.dropna()

    features = [c for c in df.columns if c not in ["timestamp", "aqi_future"]]
    X = df[features]
    y = df["aqi_future"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

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

    # Lưu mô hình
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"aqi_model_{horizon}h.pkl")
    joblib.dump(model, model_path)
    joblib.dump(features, os.path.join(MODEL_DIR, f"feature_names_{horizon}h.pkl"))

    # Lưu biểu đồ quan trọng đặc trưng
    plt.figure(figsize=(8, 6))
    lgb.plot_importance(model, max_num_features=10, importance_type="gain")
    plt.title(f"Top 10 Feature Importance (+{horizon}h)")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"feature_importance_{horizon}h.png"))
    plt.close()

    print(f"✅ AQI +{horizon}h → MAE: {mae:.3f} | R²: {r2:.3f}")
    logging.info(f"Trained model +{horizon}h | MAE={mae:.3f} | R2={r2:.3f}")

    return {"horizon": horizon, "mae": mae, "r2": r2, "model": model}


# ============================================================
#  🔁 HUẤN LUYỆN TOÀN BỘ
# ============================================================
print("\n🚀 Bắt đầu huấn luyện mô hình dự báo AQI (1h, 3h, 6h)...\n")
results = [train_for_horizon(h) for h in HORIZONS]

# ============================================================
#  📊 TỔNG KẾT
# ============================================================
print("\n" + "=" * 60)
print("📈 HIỆU SUẤT CÁC MÔ HÌNH DỰ BÁO AQI")
print("=" * 60)
for r in results:
    print(f"⏱ {r['horizon']}h → MAE: {r['mae']:.3f} | R²: {r['r2']:.3f}")
print("=" * 60)
print("✅ Huấn luyện hoàn tất.\n")
logging.info("✅ Huấn luyện hoàn tất.")
