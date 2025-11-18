import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import timedelta, datetime
from sentence_transformers import SentenceTransformer, util

# =============================
# ⚙️ CẤU HÌNH ĐƯỜNG DẪN 
# =============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(BASE_DIR, "data/raw/air_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    1: os.path.join(MODEL_DIR, "pm2_5_model_1h.pkl"), 
    3: os.path.join(MODEL_DIR, "pm2_5_model_3h.pkl"), 
    6: os.path.join(MODEL_DIR, "pm2_5_model_6h.pkl"), 
}
FEATURE_FILES = {
    1: os.path.join(MODEL_DIR, "pm2_5_features_1h.pkl"), 
    3: os.path.join(MODEL_DIR, "pm2_5_features_3h.pkl"), 
    6: os.path.join(MODEL_DIR, "pm2_5_features_6h.pkl"), 
}

# =============================
# 🔹 HÀM QUY ĐỔI PM2.5 → AQI
# =============================
def pm25_to_aqi(pm25):
    """Quy đổi giá trị PM2.5 sang chỉ số AQI (chuẩn US EPA)"""
    breakpoints = [
        (0.0, 12.0, 0, 50, "Tốt", "Không khí trong lành — rất tốt cho sức khỏe."),
        (12.1, 35.4, 51, 100, "Trung bình", "Người nhạy cảm có thể bị ảnh hưởng nhẹ."),
        (35.5, 55.4, 101, 150, "Kém", "Người già, trẻ nhỏ, người bệnh nên hạn chế ra ngoài."),
        (55.5, 150.4, 151, 200, "Xấu", "Ô nhiễm; tránh ra ngoài và hoạt động mạnh."),
        (150.5, 250.4, 201, 300, "Rất xấu", "Rất ô nhiễm; ở trong nhà nếu có thể."),
        (250.5, 500.0, 301, 500, "Nguy hại", "Không ra ngoài, nguy hiểm đến sức khỏe."),
    ]
    # Xử lý trường hợp PM2.5 rất cao
    if pm25 > 500.0:
        return 500, "Nguy hại", "Không ra ngoài, nguy hiểm đến sức khỏe."
        
    for low, high, aqi_low, aqi_high, cat, msg in breakpoints:
        if low <= pm25 <= high:
            aqi = ((aqi_high - aqi_low) / (high - low)) * (pm25 - low) + aqi_low
            return round(aqi, 1), cat, msg
    
    # Trường hợp dự phòng nếu pm25 là âm hoặc NaN
    if pd.isna(pm25) or pm25 < 0:
        return 0, "Không xác định", "Không có dữ liệu PM2.5"
        
    return 500, "Nguy hại", "Giá trị vượt ngưỡng cho phép." # Mặc định cho giá trị > 500

# =============================
# 🩺 HIỂN THỊ THẺ CẢNH BÁO
# =============================
def health_card_html(pm25):
    aqi, level, msg = pm25_to_aqi(pm25)
    color_map = {
        "Tốt": "#C8E6C9",
        "Trung bình": "#FFF9C4",
        "Kém": "#FFE0B2",
        "Xấu": "#FFCDD2",
        "Rất xấu": "#E1BEE7",
        "Nguy hại": "#B39DDB",
        "Không xác định": "#f1f5f9"
    }
    color = color_map.get(level, "#f1f5f9")
    return f"""
    <div style="background:{color}; padding:14px; border-radius:10px; margin-bottom:6px;">
      <h4 style="margin:4px 0;">AQI: <b>{aqi}</b> — {level}</h4>
      <p style="font-size:13px;margin:0;">{msg}</p>
    </div>
    """

# =============================
# 📈 HÀM XỬ LÝ DỮ LIỆU VÀ DỰ ĐOÁN 
# =============================
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
    latest = df_all.iloc[-1]
    preds = {} # Sẽ lưu PM2.5 thô
    chart_rows = []
    
    # Lấy PM2.5 lịch sử và QUY ĐỔI sang AQI
    recent = df_all.tail(24)[["timestamp", "pm2_5"]].copy()
    recent["value"] = recent["pm2_5"].apply(lambda x: pm25_to_aqi(x)[0]) # Quy đổi
    recent["type"] = "actual"
    chart_rows.extend(recent[["timestamp", "value", "type"]].to_dict(orient="records"))

    for h, model in models.items():
        feats = feature_names[h]
        df_input = build_input_df_from_latest(latest, feats)
        
        # logic lấy lag cho khớp với train_model.py
        for col in ["pm2_5", "pm10", "co", "no2", "o3", "so2", "temp", "humidity"]:
             for lag in range(1, 7): 
                feature_name = f"{col}_lag{lag}"
                if feature_name in feats:
                    try:
                        df_input[feature_name] = float(df_all[col].shift(lag).iloc[-1])
                    except Exception:
                        df_input[feature_name] = 0.0
        
        if "pm2_5_roll3" in feats: # Thêm logic rolling
            try:
                df_input["pm2_5_roll3"] = float(df_all["pm2_5"].rolling(window=3).mean().iloc[-1])
            except Exception:
                df_input["pm2_5_roll3"] = 0.0

        pm25_pred = max(0.0, model.predict(df_input[feats])[0])
        preds[h] = pm25_pred # Lưu PM2.5 thô
        
        # Quy đổi PM2.5 dự đoán sang AQI để vẽ
        aqi_pred, _, _ = pm25_to_aqi(pm25_pred)
        
        chart_rows.append({
            "timestamp": pd.to_datetime(latest["timestamp"]) + timedelta(hours=h),
            "value": aqi_pred, 
            "type": f"pred_{h}h"
        })

    chart_df = pd.DataFrame(chart_rows).sort_values("timestamp")
    return {"preds": preds, "chart_df": chart_df} # preds là PM2.5, chart_df là AQI

# =============================
# 📦 NẠP MÔ HÌNH VÀ DỮ LIỆU
# =============================
models, feature_names = {}, {}
for h in [1, 3, 6]:
    if not (os.path.exists(MODEL_FILES[h]) and os.path.exists(FEATURE_FILES[h])):
        st.error(f"❌ Thiếu mô hình {h}h ({MODEL_FILES[h]}) — hãy chạy train_model.py trước.")
        st.stop()
    models[h] = joblib.load(MODEL_FILES[h])
    feature_names[h] = joblib.load(FEATURE_FILES[h])

if not os.path.exists(DATA_PATH):
    st.error(f"❌ Không tìm thấy file dữ liệu data/raw/air_data.csv!")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    if df.empty or "pm2_5" not in df.columns or df["pm2_5"].isna().all():
        st.warning("⚠️ Dữ liệu rỗng hoặc thiếu PM2.5. Hãy chạy collector.py.")
        st.stop()
except Exception as e:
    st.error(f"Lỗi khi đọc file CSV: {e}")
    st.stop()


# =============================
# SEMANTIC MODEL (HIỂU NGỮ NGHĨA)
# =============================
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

semantic_model = load_semantic_model()

INTENTS = {
    "predict": ["dự đoán aqi", "dự báo aqi", "chất lượng không khí tương lai", "không khí vài giờ tới", "air quality forecast"],
    "chart": ["xem biểu đồ", "đồ thị aqi", "graph", "chart", "thống kê không khí"],
    "warning": ["cảnh báo", "ra đường có an toàn không", "khẩu trang", "sức khỏe", "nên ra ngoài không", "nguy hiểm không"],
    "greet": ["chào", "xin chào", "hi", "hello", "bạn là ai", "chào buổi sáng"],
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

# =============================
# UI SETUP 
# =============================
st.set_page_config(page_title="AIRCARE - Dự đoán AQI (từ PM2.5)", page_icon="🌤️", layout="wide")
st.markdown("<style> .stApp { background-color: #f8fafc; } </style>", unsafe_allow_html=True)

col1, col2 = st.columns([7, 3])
with col1:
    st.title("🌤️ AIRCARE — Dự đoán AQI (từ PM2.5)")
    st.subheader("Hỏi về chất lượng không khí hoặc dự đoán AQI (từ PM2.5) trong 1h, 3h, 6h tới.")
with col2:
    st.image("https://img.icons8.com/fluency/96/air-quality.png", width=90)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("👤 Thông tin người dùng")
    age = st.number_input("Tuổi", 1, 120, 25)
    disease = st.text_input("Bệnh lý(nếu có)", placeholder="VD: hen suyễn, viêm xoang, tim mạch...")
    if st.button("🚀 Dự đoán nhanh"):
        out = predict_multi_horizon(df, models, feature_names)
        st.session_state["last_prediction"] = out

# Khởi tạo session state
if "chat" not in st.session_state:
    st.session_state["chat"] = []

# =============================
# MAIN LAYOUT
# =============================
left, right = st.columns([3, 2])

# --- CHATBOT ---
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
            
            # Phải quy đổi PM2.5 (v) sang AQI trước khi hiển thị
            preds_text = ", ".join(
                [f"+{h}h: {pm25_to_aqi(v)[0]} AQI ({pm25_to_aqi(v)[1]})" for h, v in out["preds"].items()]
            )
            st.session_state.chat.append({"role": "bot", "text": f"🤖 Dự báo chất lượng không khí: {preds_text}"})

        elif intent == "chart":
            st.session_state.chat.append({"role": "bot", "text": "📊 Biểu đồ AQI hiển thị bên phải nhé!"})

        elif intent == "warning":
            # Phải đọc PM2.5 hiện tại và quy đổi
            latest_pm = df.iloc[-1]["pm2_5"]
            aqi, cat, msg = pm25_to_aqi(latest_pm) # Dùng hàm quy đổi
            advice = msg # Lấy thông điệp sức khỏe từ hàm
            
            if disease:
                advice += f" Vì bạn có bệnh '{disease}', nên đặc biệt chú ý khi AQI > 100."
            
            # Trả về thông tin đã quy đổi
            st.session_state.chat.append({"role": "bot", "text": f"AQI hiện tại {aqi:.0f} — {cat}. {advice}"})

        elif intent == "greet":
            st.session_state.chat.append({"role": "bot", "text": "Chào👋! Tôi là AirCare Chatbot, sẵn sàng giúp bạn theo dõi chất lượng không khí 🌤️."})

        else:
            st.session_state.chat.append({"role": "bot", "text": "🤔 Mình chưa hiểu rõ ý bạn. Hãy thử hỏi: 'Dự đoán AQI', 'Biểu đồ AQI' hoặc 'Cảnh báo'."})

        st.session_state.input_text = ""

    st.text_input("Nhập tin nhắn...", key="input_text", on_change=handle_message)
    
# --- BIỂU ĐỒ & CẢNH BÁO ---
with right:
    st.subheader("📊 Dự báo & Cảnh báo")
    latest = df.iloc[-1]
    aqi_now, level, msg = pm25_to_aqi(latest["pm2_5"])
    st.write(f"⏰ {latest['timestamp']} — AQI hiện tại: {aqi_now:.0f} ({level})")
    st.info(msg) # Hiển thị thông báo sức khỏe hiện tại

    if "last_prediction" in st.session_state:
        out = st.session_state["last_prediction"]
        # Thẻ cảnh báo (Health card)
        for h, pm25_val in out["preds"].items():
            st.markdown(health_card_html(pm25_val), unsafe_allow_html=True)

        # Biểu đồ này hiện đã vẽ AQI 
        chart_df = out["chart_df"]
        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="Thời gian"), 
                y=alt.Y("value:Q", title="Chỉ số AQI"), 
                color=alt.Color("type:N", title="Loại"),
                tooltip=["timestamp:T", "value:Q", "type:N"] 
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Nhấn **🚀 Dự đoán nhanh** hoặc hỏi chatbot để xem dự báo.")