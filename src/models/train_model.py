import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ============================================================
# ⚙️ CẤU HÌNH ĐƯỜNG DẪN VÀ LOGIC THỜI GIAN 
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(BASE_DIR, "data/raw/air_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
HORIZONS = [1, 3, 6]  # dự báo 1h, 3h, 6h tới
os.makedirs(MODEL_DIR, exist_ok=True)

# Dữ liệu được thu thập mỗi 2 phút (theo file collector.py)
INTERVAL_MINUTES = 2
ROWS_PER_HOUR = 60 // INTERVAL_MINUTES  # 30 hàng = 1 giờ

# ============================================================
# 📥 NẠP DỮ LIỆU
# ============================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại {DATA_PATH}")

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
df = df.dropna(subset=["pm2_5"]) 
print(f"✅ Nạp dữ liệu: {len(df)} dòng, {df['timestamp'].min()} → {df['timestamp'].max()}")

# ============================================================
# 🧩 TẠO FEATURE CHO PM2.5 
# ============================================================
def create_features(df, shift_rows): 
    """Tạo feature và target cho từng horizon."""
    df_feat = df.copy()
    
    # Thông tin thời gian
    ts = df_feat["timestamp"]
    df_feat["hour"] = ts.dt.hour
    df_feat["weekday"] = ts.dt.weekday
    df_feat["month"] = ts.dt.month
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24)

    # Lag features cho pm2_5 và các khí khác
    for col in ["pm2_5", "pm10", "co", "no2", "o3", "so2", "temp", "humidity"]:
        if col in df_feat.columns: 
            for lag in range(1, 7):  
                df_feat[f"{col}_lag{lag}"] = df_feat[col].shift(lag)

    # Rolling mean
    df_feat["pm2_5_roll3"] = df_feat["pm2_5"].rolling(window=3).mean()

    # Target: shift tương ứng SỐ HÀNG (ví dụ: 30, 90, 180 hàng)
    df_feat["pm2_5_target"] = df_feat["pm2_5"].shift(-shift_rows) 
    df_feat = df_feat.dropna().reset_index(drop=True)

    # Chọn feature columns (loại bỏ các cột không phải là feature)
    cols_to_drop = ["timestamp", "pm2_5_target", "aqi", "aqi_level"]
    feature_cols = [
        c for c in df_feat.columns if c not in cols_to_drop and c in df.columns
    ]
    
    X = df_feat[feature_cols]
    y = df_feat["pm2_5_target"]
    return X, y, feature_cols

# ============================================================
# 🧠 HUẤN LUYỆN VÀ LƯU MÔ HÌNH 
# ============================================================
def train_for_horizon(horizon_in_hours): 
    shift_rows = horizon_in_hours * ROWS_PER_HOUR  

    # Kiểm tra xem có đủ dữ liệu để shift không
    if len(df) <= shift_rows:
        print(f"❌ Không đủ dữ liệu (cần > {shift_rows} dòng) để huấn luyện cho {horizon_in_hours}h. Bỏ qua.")
        return None

    X, y, features = create_features(df, shift_rows=shift_rows) 

    # Kiểm tra lại sau khi dropna
    if X.empty:
        print(f"❌ Không còn dữ liệu sau khi xử lý feature cho {horizon_in_hours}h. Bỏ qua.")
        return None

    # Chia train/test theo thời gian
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
        "verbose": -1
    }

    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_test, y_test, reference=train_data)

    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    # Đánh giá
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model_path = os.path.join(MODEL_DIR, f"pm2_5_model_{horizon_in_hours}h.pkl")
    features_path = os.path.join(MODEL_DIR, f"pm2_5_features_{horizon_in_hours}h.pkl")
    joblib.dump(model, model_path)
    joblib.dump(features, features_path)

    print(f"✅ PM2.5 +{horizon_in_hours}h → MAE: {mae:.3f} | R²: {r2:.3f} | 📁 {os.path.basename(model_path)}")
    return {"horizon": horizon_in_hours, "mae": mae, "r2": r2}

# ============================================================
# 🚀 HUẤN LUYỆN TOÀN BỘ CÁC HORIZONS
# ============================================================
print("\n🚀 Bắt đầu huấn luyện mô hình PM2.5 (1h, 3h, 6h)...\n")
results = [train_for_horizon(h) for h in HORIZONS]
results = [r for r in results if r is not None] # Lọc bỏ các lần chạy bị lỗi (do thiếu data)

# ============================================================
# 📊 TỔNG KẾT
# ============================================================
print("\n" + "=" * 60)
print("📈 HIỆU SUẤT CÁC MÔ HÌNH DỰ BÁO PM2.5")
print("=" * 60)
if not results:
    print("Không có mô hình nào được huấn luyện thành công (có thể do thiếu dữ liệu).")
else:
    for r in results:
        print(f"⏱ {r['horizon']}h → MAE: {r['mae']:.3f} | R²: {r['r2']:.3f}")
print("=" * 60)
print("✅ Huấn luyện hoàn tất.\n")