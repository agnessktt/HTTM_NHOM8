import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import timedelta
from sentence_transformers import SentenceTransformer, util

# =============================
# 🔧 CẤU HÌNH ĐƯỜNG DẪN
# =============================
DATA_PATH = "data/raw/air_data.csv"
MODEL_DIR = "models"

MODEL_FILES = {
    1: f"{MODEL_DIR}/aqi_model_1h.pkl",
    3: f"{MODEL_DIR}/aqi_model_3h.pkl",
    6: f"{MODEL_DIR}/aqi_model_6h.pkl",
}
FEATURE_FILES = {
    1: f"{MODEL_DIR}/feature_names_1h.pkl",
    3: f"{MODEL_DIR}/feature_names_3h.pkl",
    6: f"{MODEL_DIR}/feature_names_6h.pkl",
}


# =============================
# 🧠 HÀM TIỆN ÍCH
# =============================

def health_card_html(aqi):
    """Trả về HTML cảnh báo màu tương ứng mức AQI"""
    if aqi <= 50:
        level, color, msg = "Tốt", "#C8E6C9", "Không khí trong lành — rất tốt cho sức khỏe."
    elif aqi <= 100:
        level, color, msg = "Trung bình", "#FFF9C4", "Chất lượng tạm ổn; người nhạy cảm chú ý."
    elif aqi <= 150:
        level, color, msg = "Kém", "#FFE0B2", "Không tốt cho người nhạy cảm; nên hạn chế ra ngoài."
    elif aqi <= 200:
        level, color, msg = "Xấu", "#FFCDD2", "Ô nhiễm; tránh ra ngoài và hoạt động mạnh."
    elif aqi <= 300:
        level, color, msg = "Rất xấu", "#E1BEE7", "Rất ô nhiễm; hạn chế tối đa ra ngoài."
    else:
        level, color, msg = "Nguy hại", "#B39DDB", "Nguy hiểm — không ra ngoài nếu không cần thiết."

    return f"""
    <div style="background:{color}; padding:14px; border-radius:10px; margin-bottom:6px;">
      <h4 style="margin:4px 0;">AQI: <b>{aqi:.0f}</b> — {level}</h4>
      <p style="font-size:13px;margin:0;">{msg}</p>
    </div>
    """


def build_input_df_from_latest(latest_row, feature_list):
    """Tạo DataFrame đầu vào từ hàng dữ liệu mới nhất"""
    base = {k: float(latest_row.get(k, 0)) for k in
            ["pm2_5", "pm10", "no2", "co", "o3", "so2", "temp", "humidity"]}
    ts = pd.to_datetime(latest_row["timestamp"])
    base.update({
        "hour": ts.hour,
        "weekday": ts.weekday(),
        "month": ts.month,
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24)
    })

    df_in = pd.DataFrame([base])
    for f in feature_list:
        if f not in df_in.columns:
            df_in[f] = 0.0
    return df_in[feature_list]


def predict_multi_horizon(df_all, models, feature_names):
    """Dự đoán AQI cho các mốc 1h, 3h, 6h"""
    latest = df_all.iloc[-1]
    preds, chart_rows = {}, []

    # Dữ liệu thực tế 24h gần nhất
    recent = df_all.tail(24)[["timestamp", "aqi"]].copy()
    recent["type"] = "actual"
    chart_rows.extend(recent.to_dict(orient="records"))

    # Dự đoán cho các khoảng thời gian
    for h, model in models.items():
        feats = feature_names[h]
        df_input = build_input_df_from_latest(latest, feats)
        for col in feats:
            if col.startswith("aqi_lag"):
                try:
                    lag_n = int(col.split("aqi_lag")[-1])
                    df_input[col] = float(df_all["aqi"].shift(lag_n).iloc[-1])
                except Exception:
                    df_input[col] = 0.0

        pred = max(0.0, model.predict(df_input)[0])
        preds[h] = pred
        chart_rows.append({
            "timestamp": pd.to_datetime(latest["timestamp"]) + timedelta(hours=h),
            "aqi": pred,
            "type": f"pred_{h}h"
        })

    chart_df = pd.DataFrame(chart_rows).sort_values("timestamp")
    return {"preds": preds, "chart_df": chart_df}


# =============================
# 📦 LOAD MÔ HÌNH & DỮ LIỆU
# =============================

models, feature_names = {}, {}
for h in [1, 3, 6]:
    if not (os.path.exists(MODEL_FILES[h]) and os.path.exists(FEATURE_FILES[h])):
        st.error(f"❌ Thiếu mô hình {h}h — hãy chạy train_model.py trước.")
        st.stop()
    models[h] = joblib.load(MODEL_FILES[h])
    feature_names[h] = joblib.load(FEATURE_FILES[h])

if not os.path.exists(DATA_PATH):
    st.error("❌ Không tìm thấy file dữ liệu!")
    st.stop()

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
if df.empty:
    st.warning("⚠️ Dữ liệu rỗng. Hãy kiểm tra lại.")
    st.stop()


# =============================
# 🤖 MÔ HÌNH NGỮ NGHĨA
# =============================

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


semantic_model = load_semantic_model()

INTENTS = {
    "predict": ["dự đoán aqi", "dự báo aqi", "chất lượng không khí tương lai", "không khí vài giờ tới"],
    "chart": ["xem biểu đồ", "đồ thị aqi", "graph", "chart", "thống kê không khí"],
    "warning": ["cảnh báo", "ra đường có an toàn không", "khẩu trang", "sức khỏe", "nên ra ngoài không"],
    "greet": ["chào", "xin chào", "hello", "bạn là ai"],
}


def detect_intent(user_text):
    """Xác định ý định người dùng dựa trên độ tương đồng ngữ nghĩa"""
    user_emb = semantic_model.encode(user_text, convert_to_tensor=True)
    best_intent, best_score = "unknown", 0.0
    for intent, examples in INTENTS.items():
        intent_emb = semantic_model.encode(examples, convert_to_tensor=True)
        sim = util.cos_sim(user_emb, intent_emb).max().item()
        if sim > best_score:
            best_score, best_intent = sim, intent
    return best_intent if best_score > 0.55 else "unknown"


# =============================
# 🎨 GIAO DIỆN STREAMLIT
# =============================

st.set_page_config(page_title="AIRCARE Chatbot AQI", page_icon="🌤️", layout="wide")
st.markdown("<style>.stApp { background-color: #f8fafc; }</style>", unsafe_allow_html=True)

col1, col2 = st.columns([7, 3])
with col1:
    st.title("🌤️ AIRCARE — Chatbot Dự đoán AQI")
    st.caption("Hỏi về chất lượng không khí hoặc dự đoán AQI trong 1h, 3h, 6h tới.")
with col2:
    st.image("https://img.icons8.com/fluency/96/air-quality.png", width=90)
st.markdown("---")

# Sidebar: thông tin người dùng
with st.sidebar:
    st.header("👤 Thông tin người dùng")
    age = st.number_input("Tuổi", 1, 120, 25)
    disease = st.text_input("Bệnh lý (nếu có)", placeholder="VD: hen suyễn, viêm xoang, tim mạch...")
    if st.button("🚀 Dự đoán nhanh"):
        out = predict_multi_horizon(df, models, feature_names)
        st.session_state["last_prediction"] = out

# Khởi tạo session state
if "chat" not in st.session_state:
    st.session_state["chat"] = []


# =============================
# 💬 CHATBOT & BIỂU ĐỒ
# =============================

left, right = st.columns([3, 2])

# --- CHAT ---
with left:
    st.subheader("💬 Chatbot")
    for msg in st.session_state["chat"]:
        align = "right" if msg["role"] == "user" else "left"
        color = "#0b5cff" if msg["role"] == "user" else "#111"
        bg = "#dbeafe" if msg["role"] == "user" else "#f1f5f9"
        st.markdown(f"""
            <div style="text-align:{align}; background:{bg};
            padding:8px; border-radius:8px; margin-bottom:6px; color:{color}">
            <b>{'Bạn' if msg['role']=='user' else 'Bot'}:</b> {msg['text']}
            </div>
        """, unsafe_allow_html=True)

    def handle_message():
        user_text = st.session_state.input_text.strip()
        if not user_text:
            return
        st.session_state.chat.append({"role": "user", "text": user_text})

        intent = detect_intent(user_text.lower())

        if intent == "predict":
            out = predict_multi_horizon(df, models, feature_names)
            st.session_state["last_prediction"] = out
            preds_text = ", ".join([f"+{h}h: {v:.0f}" for h, v in out["preds"].items()])
            reply = f"🤖 Dự báo AQI: {preds_text}"
        elif intent == "chart":
            reply = "📊 Biểu đồ AQI hiển thị ở khung bên phải nhé!"
        elif intent == "warning":
            latest = df.iloc[-1]["aqi"]
            if latest > 150:
                msg = "⚠️ Không khí ô nhiễm, hạn chế ra ngoài."
            elif latest > 100:
                msg = "😷 Không khí trung bình, người nhạy cảm nên đeo khẩu trang."
            else:
                msg = "✅ Không khí trong lành, bạn có thể ra ngoài thoải mái."
            if disease:
                msg += f" Vì bạn có bệnh '{disease}', nên chú ý hơn khi AQI >100."
            reply = msg
        elif intent == "greet":
            reply = "Chào 👋! Tôi là AirCare Chatbot, giúp bạn theo dõi chất lượng không khí 🌤️."
        else:
            reply = "🤔 Mình chưa hiểu rõ ý bạn. Hãy thử hỏi: 'Dự đoán AQI', 'Biểu đồ AQI' hoặc 'Cảnh báo'."

        st.session_state.chat.append({"role": "bot", "text": reply})
        st.session_state.input_text = ""

    st.text_input("Nhập tin nhắn...", key="input_text", on_change=handle_message)


# --- BIỂU ĐỒ ---
with right:
    st.subheader("📊 Dự đoán & Cảnh báo")
    latest = df.iloc[-1]
    st.write(f"⏰ {latest['timestamp']} — AQI hiện tại: {latest['aqi']:.0f}")

    if "last_prediction" in st.session_state:
        out = st.session_state["last_prediction"]
        for h, aqi in out["preds"].items():
            st.markdown(health_card_html(aqi), unsafe_allow_html=True)

        chart_df = out["chart_df"]
        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(x="timestamp:T", y="aqi:Q", color="type:N")
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Nhấn **🚀 Dự đoán nhanh** hoặc hỏi chatbot để xem dự báo.")
