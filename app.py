# ==================================================
# 🌤️ AIRCARE PRO — CHATBOT DỰ ĐOÁN AQI NÂNG CẤP
# ==================================================
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import timedelta, datetime
from sentence_transformers import SentenceTransformer, util

# ==============================
# CẤU HÌNH
# ==============================
DATA_PATH = "data/raw/air_data.csv"
MODEL_DIR = "models"
MODEL_FILES = {h: f"{MODEL_DIR}/aqi_model_{h}h.pkl" for h in [1,3,6]}
FEATURE_FILES = {h: f"{MODEL_DIR}/feature_names_{h}h.pkl" for h in [1,3,6]}
LOG_PATH = "data/logs/user_interactions.csv"
REFRESH_INTERVAL = 10  # phút

# ==============================
# TIỆN ÍCH
# ==============================
def health_card_html(aqi):
    """Cảnh báo màu theo mức AQI"""
    levels = [
        (50, "Tốt", "#C8E6C9", "Không khí trong lành — rất tốt cho sức khỏe."),
        (100, "Trung bình", "#FFF9C4", "Chất lượng tạm ổn; người nhạy cảm chú ý."),
        (150, "Kém", "#FFE0B2", "Không tốt cho người nhạy cảm; nên hạn chế ra ngoài."),
        (200, "Xấu", "#FFCDD2", "Ô nhiễm; tránh ra ngoài và hoạt động mạnh."),
        (300, "Rất xấu", "#E1BEE7", "Rất ô nhiễm; hạn chế tối đa ra ngoài."),
    ]
    for limit, level, color, msg in levels:
        if aqi <= limit:
            break
    else:
        level, color, msg = "Nguy hại", "#B39DDB", "Nguy hiểm — không ra ngoài nếu không cần thiết."

    return f"""
    <div style="background:{color}; padding:14px; border-radius:10px; margin-bottom:6px;">
      <h4 style="margin:4px 0;">AQI: <b>{aqi:.0f}</b> — {level}</h4>
      <p style="font-size:13px;margin:0;">{msg}</p>
    </div>
    """

def build_input_df_from_latest(latest_row, feature_list):
    base = {k: float(latest_row.get(k, 0)) for k in ["pm2_5","pm10","no2","co","o3","so2","temp","humidity"]}
    ts = pd.to_datetime(latest_row["timestamp"])
    base.update({
        "hour": ts.hour,
        "weekday": ts.weekday(),
        "month": ts.month,
        "hour_sin": np.sin(2*np.pi*ts.hour/24),
        "hour_cos": np.cos(2*np.pi*ts.hour/24)
    })
    df_in = pd.DataFrame([base])
    for f in feature_list:
        if f not in df_in.columns:
            df_in[f] = 0.0
    return df_in[feature_list]

def predict_multi_horizon(df_all, models, feature_names):
    """Dự đoán AQI +1h, +3h, +6h"""
    latest = df_all.iloc[-1]
    preds, chart_rows = {}, []
    recent = df_all.tail(24)[["timestamp", "aqi"]].copy()
    recent["type"] = "actual"
    chart_rows.extend(recent.to_dict(orient="records"))

    for h, model in models.items():
        feats = feature_names[h]
        df_input = build_input_df_from_latest(latest, feats)
        for col in feats:
            if col.startswith("aqi_lag"):
                try:
                    lag_n = int(col.split("aqi_lag")[-1])
                    df_input[col] = float(df_all["aqi"].shift(lag_n).iloc[-1])
                except:
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

def log_interaction(user_text, intent):
    """Ghi log người dùng"""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()},{user_text},{intent}\n")

# ==============================
# LOAD MODEL
# ==============================
models, feature_names = {}, {}
for h in [1,3,6]:
    models[h] = joblib.load(MODEL_FILES[h])
    feature_names[h] = joblib.load(FEATURE_FILES[h])

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")

# ==============================
# SEMANTIC CHATBOT
# ==============================
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

semantic_model = load_semantic_model()
INTENTS = {
    "predict": ["dự đoán aqi", "dự báo không khí", "air forecast"],
    "chart": ["xem biểu đồ", "đồ thị aqi", "graph", "chart"],
    "warning": ["cảnh báo", "ra ngoài có an toàn không", "khẩu trang", "sức khỏe"],
    "weather": ["nhiệt độ", "độ ẩm", "thời tiết"],
    "greet": ["chào", "hi", "hello"],
}

def detect_intent(user_text):
    user_emb = semantic_model.encode(user_text, convert_to_tensor=True)
    best_intent, best_score = "unknown", 0.0
    for intent, examples in INTENTS.items():
        intent_emb = semantic_model.encode(examples, convert_to_tensor=True)
        sim = util.cos_sim(user_emb, intent_emb).max().item()
        if sim > best_score:
            best_score, best_intent = sim, intent
    return best_intent if best_score > 0.55 else "unknown"

# ==============================
# GIAO DIỆN STREAMLIT
# ==============================
st.set_page_config(page_title="AIRCARE PRO", page_icon="🌤️", layout="wide")

col1, col2 = st.columns([7,3])
with col1:
    st.title("🌤️ AIRCARE PRO — Chatbot Dự đoán AQI")
    st.caption("Dự báo AQI, phân tích rủi ro sức khỏe và xem biểu đồ chất lượng không khí.")
with col2:
    st.image("https://img.icons8.com/fluency/96/air-quality.png", width=90)

# Sidebar
with st.sidebar:
    st.header("👤 Thông tin người dùng")
    age = st.number_input("Tuổi", 1, 120, 25)
    disease = st.text_input("Bệnh lý (nếu có)", placeholder="VD: hen suyễn, tim mạch...")
    refresh = st.checkbox("Tự động làm mới dữ liệu mỗi 10 phút", True)
    if st.button("🚀 Dự đoán nhanh"):
        st.session_state["last_prediction"] = predict_multi_horizon(df, models, feature_names)

# ==============================
# CHATBOT
# ==============================
if "chat" not in st.session_state: st.session_state["chat"] = []

def handle_message():
    user_text = st.session_state.input_text.strip()
    if not user_text: return
    st.session_state.chat.append({"role": "user", "text": user_text})

    intent = detect_intent(user_text.lower())
    log_interaction(user_text, intent)

    if intent == "predict":
        out = predict_multi_horizon(df, models, feature_names)
        st.session_state["last_prediction"] = out
        preds = ", ".join([f"+{h}h: {v:.0f}" for h,v in out["preds"].items()])
        reply = f"🤖 Dự báo AQI: {preds}"
    elif intent == "chart":
        reply = "📊 Biểu đồ hiển thị bên phải nhé!"
    elif intent == "warning":
        aqi = df.iloc[-1]["aqi"]
        if aqi > 150:
            reply = "⚠️ Không khí ô nhiễm, nên ở trong nhà."
        elif aqi > 100:
            reply = "😷 Không khí trung bình, nhớ đeo khẩu trang khi ra ngoài."
        else:
            reply = "✅ Không khí trong lành, bạn có thể ra ngoài thoải mái."
        if disease:
            reply += f" Do bạn có bệnh '{disease}', hãy chú ý khi AQI >100."
    elif intent == "weather":
        latest = df.iloc[-1]
        reply = f"🌡️ Nhiệt độ: {latest['temp']}°C, 💧 Độ ẩm: {latest['humidity']}%."
    elif intent == "greet":
        reply = "Chào bạn 👋, tôi là AirCare Bot — sẵn sàng giúp bạn theo dõi không khí!"
    else:
        reply = "🤔 Mình chưa hiểu rõ ý bạn. Hãy thử hỏi: 'Dự đoán AQI' hoặc 'Thời tiết'."

    st.session_state.chat.append({"role": "bot", "text": reply})
    st.session_state.input_text = ""

# Chat UI
st.text_input("Nhập tin nhắn...", key="input_text", on_change=handle_message)
for msg in st.session_state["chat"]:
    align = "right" if msg["role"] == "user" else "left"
    bg = "#dbeafe" if msg["role"] == "user" else "#f1f5f9"
    st.markdown(f"<div style='text-align:{align}; background:{bg}; padding:8px; border-radius:8px; margin-bottom:6px;'>{msg['text']}</div>", unsafe_allow_html=True)

# ==============================
# BIỂU ĐỒ & CẢNH BÁO
# ==============================
st.markdown("---")
st.subheader("📊 Dự đoán & Phân tích")
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
        .encode(
            x="timestamp:T",
            y=alt.Y("aqi:Q", title="Chỉ số AQI"),
            color=alt.Color("type:N", title="Loại dữ liệu"),
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("💡 Nhấn **🚀 Dự đoán nhanh** hoặc hỏi chatbot để xem dự báo.")
