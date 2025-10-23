import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.model.train_model import train_aqi_model

st.set_page_config(page_title="AirCare AQI Monitor", layout="wide")

st.title("🌤️ AirCare — Air Quality Index (AQI) Monitoring Dashboard")
st.markdown("Theo dõi và dự đoán chất lượng không khí tại **Hà Nội** theo thời gian thực.")

# --- Nút load dữ liệu ---
data_path = "data/raw/air_data.csv"

if st.button("🔄 Làm mới dữ liệu"):
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        st.success("✅ Dữ liệu mới nhất đã được tải lại!")
    else:
        st.error("❌ Không tìm thấy file dữ liệu!")
else:
    # Lần đầu load
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        st.error("❌ Không tìm thấy file dữ liệu!")
        st.stop()

if not data.empty:
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")

    # --- Hiển thị dữ liệu mới nhất ---
    st.subheader("📊 Dữ liệu mới nhất từ cảm biến / API")
    st.dataframe(data.tail(5))

    # --- Biểu đồ AQI theo thời gian ---
    st.subheader("📈 Biểu đồ AQI theo thời gian")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["timestamp"], data["aqi"], marker='o', linestyle='-', color='skyblue', label="AQI thực tế")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Chỉ số AQI")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # --- Huấn luyện mô hình ---
    st.markdown("---")
    st.subheader("🤖 Huấn luyện mô hình dự đoán AQI")

    if st.button("🚀 Bắt đầu huấn luyện"):
        with st.spinner("Đang huấn luyện mô hình, vui lòng đợi... ⏳"):
            model, metrics = train_aqi_model()
        st.success("✅ Huấn luyện hoàn tất!")

        st.markdown("### 🔍 Kết quả mô hình")
        st.json(metrics)

        # --- Dự đoán ---
        feature_cols = ['pm2_5', 'pm10', 'no2', 'co', 'o3', 'so2', 'temp', 'humidity']
        predictions = model.predict(data[feature_cols])
        data['Predicted_AQI'] = predictions

        # --- Biểu đồ so sánh ---
        st.markdown("### 🔮 Biểu đồ so sánh AQI thực tế và dự đoán")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(data["timestamp"], data["aqi"], marker='o', color='blue', label='Thực tế')
        ax2.plot(data["timestamp"], data["Predicted_AQI"], marker='x', color='orange', label='Dự đoán')
        ax2.set_xlabel("Thời gian")
        ax2.set_ylabel("AQI")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

else:
    st.error("❌ Không có dữ liệu trong file CSV.")

st.markdown("---")
st.caption("© 2025 — AirCare")
