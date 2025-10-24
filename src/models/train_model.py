advice.append("⚠️ Bạn thuộc nhóm người cao tuổi, nên đặc biệt lưu ý khi AQI vượt 100.<br>")
        if user_info.get("disease"):
            disease = user_info["disease"].lower()
            if any(x in disease for x in ["hen", "xoang", "phổi", "viêm mũi", "copd"]):
                advice.append("🫁 Bạn có bệnh về <b>đường hô hấp</b>, nên hạn chế ra ngoài.<br>")
            elif any(x in disease for x in ["tim", "huyết áp", "mạch", "thiếu máu"]):
                advice.append("❤️ Bạn có bệnh về <b>tim mạch</b>, tránh vận động mạnh ngoài trời.<br>")
            elif any(x in disease for x in ["da", "mắt"]):
                advice.append("👁️ Bạn có bệnh về <b>da hoặc mắt</b>, nên tránh tiếp xúc lâu với không khí ô nhiễm.<br>")

    advice.append("<br><em>ℹ️ Thông tin chỉ mang tính tham khảo, không thay thế chẩn đoán y tế.</em>")

    return f"""
    <div style="background-color:{color}; border-radius:12px; padding:18px 22px; line-height:1.7; font-size:16px;">
        {''.join(advice)}
    </div>
    """

# -----------------------------
# Giao diện Streamlit
st.set_page_config(page_title="AIRCARE - Dự đoán AQI & Khuyến nghị sức khỏe", page_icon="🌤️", layout="centered")

st.title("🌤️ Dự đoán chỉ số AQI & Khuyến nghị sức khỏe cá nhân")

st.subheader("👤 Thông tin người dùng")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Tuổi", min_value=1, max_value=120, value=25)
with col2:
    disease = st.text_input("Bệnh lý (nếu có)", placeholder="Ví dụ: hen suyễn, xoang mũi,...")

user_info = {"age": age, "disease": disease.strip()}

# -----------------------------
# Lấy dữ liệu môi trường mới nhất
if not os.path.exists(DATA_PATH):
    st.error(f"❌ Không tìm thấy file dữ liệu tại `{DATA_PATH}`.")
    st.stop()

df = pd.read_csv(DATA_PATH)
if df.empty:
    st.warning("⚠️ Dữ liệu rỗng. Hãy cập nhật dữ liệu môi trường trước.")
    st.stop()

latest = df.iloc[-1][["pm2_5", "pm10", "no2", "co", "o3", "so2", "temp", "humidity"]].to_dict()

# -----------------------------
# Dự đoán AQI
if st.button("🚀 Dự đoán AQI"):
    # Tạo DataFrame đầu vào
    df_input = pd.DataFrame([latest])

    # Đảm bảo có đủ các cột mô hình yêu cầu
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0  # giá trị mặc định cho các feature lag/time

    # Sắp xếp đúng thứ tự cột
    df_input = df_input[feature_names]

    # Dự đoán
    try:
        aqi_pred = model.predict(df_input)[0]
        alert_html = generate_health_alert(aqi_pred, user_info)
        st.markdown(alert_html, unsafe_allow_html=True)

        with st.expander("📋 Xem dữ liệu môi trường gần nhất"):
            st.dataframe(df.tail(5))

    except Exception as e:
        st.error(f"⚠️ Lỗi khi dự đoán: {e}")
