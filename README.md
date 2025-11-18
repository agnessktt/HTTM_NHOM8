Dự án: AirCare — Dự đoán chất lượng không khí & cảnh báo sức khỏe cá nhân
---------------------------------
Thành viên trong nhóm 8
- Lê Thị Kiều Trang
- Quách Hữu Nam
- Nguyễn Minh Sang
- Nguyễn Đức Trường

---------------

1. Tóm tắt dự án
AIRCARE là một hệ thống dự báo và cảnh báo chất lượng không khí (AQI) theo thời gian thực, được xây dựng như một ML Application (Ứng dụng Học máy).

Dự án bao gồm:

Bộ thu thập dữ liệu tự động từ OpenWeather API (mỗi 2 phút), thu thập các chỉ số môi trường như PM2.5, nhiệt độ, độ ẩm....

Mô hình LightGBM được huấn luyện để dự đoán nồng độ PM2.5 (µg/m³) trong 1h, 3h và 6h tiếp theo.

Ứng dụng Streamlit Chatbot sử dụng AI Ngữ nghĩa (SentenceTransformer) để hiểu ý định người dùng, sau đó quy đổi kết quả dự đoán PM2.5 sang chỉ số AQI trực quan để cảnh báo.

2. 🗂️ Cấu trúc thư mục dự án
AIRCARE/
│
├── data/                         ← Thư mục dữ liệu
│   ├── raw/                      ←   Dữ liệu gốc (chưa xử lý)
│   │   └── air_data.csv          ←   File dữ liệu AQI gốc (các thông số môi trường)
│   └── collector.log             ←   File log khi thu thập dữ liệu
│
├── models/                       
│   ├── pm2_5_model_1h.pkl           
│   ├── pm2_5_model_3h.pkl   	
│   ├── pm2_5_model_6h.pkl         
│   ├── pm2_5_features_1h.pkl      
│   ├── pm2_5_features_3h.pkl 	
│   └── pm2_5_features_6h.pkl 	← Các mô hình LightGBM đã huấn luyện
│
├── src/
│   ├── collector.py                ← Thu thập dữ liệu AQI và thời tiết từ OpenWeather API
│   ├── train_model.py              ← Huấn luyện mô hình LightGBM cho 1h/3h/6h dự báo
│   └── run/
│       └── app.py                  ← Ứng dụng Streamlit Chatbot chính
│
├── .venv312/                       ← Môi trường ảo (virtual environment)
│
├── README.md                       ← Hướng dẫn cài đặt và chạy dự án
└── requirements.txt                ← Danh sách thư viện cần thiết


3. 🧩 Các tính năng nổi bật

🕒 Cập nhật dữ liệu tự động từ API (mỗi 2 phút). 

💬 Chatbot thông minh hiểu ngôn ngữ tự nhiên tiếng Việt (sử dụng SentenceTransformer).

📈 Biểu đồ trực quan về AQI (đã quy đổi) hiện tại và dự báo 1h, 3h, 6h tới. 

🧠 Dự báo đa thời điểm (1h, 3h, 6h) bằng mô hình LightGBM đã huấn luyện. 

❤️ Khuyến nghị sức khỏe cá nhân hóa dựa trên tuổi và bệnh lý người dùng nhập vào.

4. Cách sử dụng 

-Chạy ứng dụng: mở terminal → gõ
.\.venv312\Scripts\Activate.ps1    #bật môi trường ảo

python src/data/collector.py       #thu thập dữ liệu

python src/models/train_model.py   #huấn luyện mô hình

streamlit run src/run/app.py       #chạy giao diện

Nhập thông tin cá nhân: tuổi, tình trạng sức khỏe.

Ứng dụng tự động lấy dữ liệu PM2.5 hiện tại (từ air_data.csv) và quy đổi sang AQI để hiển thị.

Xem kết quả dự đoán PM2.5 đã được quy đổi sang AQI cho 1h, 3h, 6h tới (hiển thị bằng biểu đồ và màu cảnh báo).

Chat với chatbot để nhận lời khuyên sức khỏe tương ứng (ví dụ: "cảnh báo", "nên ra ngoài không?").

5. 🔗 Liên kết GitHub dự án

👉 Dự án đầy đủ: https://github.com/agnessktt/HTTM_NHOM8

*Note:

(Virtual Environment): Thư mục .venv được tạo cục bộ để quản lý thư viện và không được đẩy lên GitHub.

(API Key): Khóa API (OPENWEATHER_API_KEY) được lưu trong file .env (không công khai trên GitHub để bảo mật).

(Mô hình .pkl): Các file .pkl trong thư mục models/ là kết quả của quá trình huấn luyện mô hình dự đoán nồng độ PM2.5 (thay vì AQI) ở các mốc thời gian khác nhau (1h, 3h, 6h).

(log): File collector.log là file log được tạo trong quá trình chạy ứng dụng để ghi lại các thông tin thu thập dữ liệu. File này không được đẩy lên GitHub và chỉ tồn tại cục bộ trên máy khi chạy app.   