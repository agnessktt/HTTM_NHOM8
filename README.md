Dự án: AirCare — Dự đoán chất lượng không khí & cảnh báo sức khỏe cá nhân
---------------------------------
Thành viên trong nhóm 8
- Lê Thị Kiều Trang
- Quách Hữu Nam
- Nguyễn Minh Sang
- Nguyễn Đức Trường

---------------

1. Tóm tắt dự án
AIRCARE là một hệ thống dự báo và cảnh báo chất lượng không khí (AQI) theo thời gian thực.  
Dự án bao gồm:
- Bộ thu thập dữ liệu tự động từ OpenWeather API (mỗi 2 phút).
- Mô hình LightGBM dự đoán AQI trong 1h, 3h và 6h tiếp theo.
- Ứng dụng Streamlit Chatbot giúp người dùng hỏi đáp và theo dõi chất lượng không khí trực quan.

2. 🗂️ Cấu trúc thư mục dự án
AIRCARE/
|
|--data/                                <- Thư mục dữ liệu
|  |--raw/                              <- Dữ liệu gốc (chưa xử lý)
|  |    |--air_data.csv                 <- File dữ liệu AQI gốc (các thông số môi trường)
|  |--collector.log                     <- File log khi thu thập dữ liệu
|
|--models/
|  |--aqi_model_1h.pkl
|  |--aqi_model_3h.pkl
|  |--aqi_model_6h.pkl
|  |--feature_names_1h.pkl
|  |--feature_names_3h.pkl
|  |--feature_names_6h.pkl              <- Các mô hình LightGBM đã huấn luyện
|
|--src/
|  |--collector.py                      <- Thu thập dữ liệu AQI và thời tiết từ OpenWeather API
|  |--train_model.py                    <- Huấn luyện mô hình LightGBM cho 1h/3h/6h dự báo
|  |--run/
|     |--app.py                         <- Ứng dụng Streamlit Chatbot chính
|
|--.venv312/                            <- Môi trường ảo (virtual environment)
|
|--README.md                            <- Hướng dẫn cài đặt và chạy dự án
|--requirements.txt                     <- Danh sách thư viện cần thiết

3. 🧩 Các tính năng nổi bật

🕒 Cập nhật dữ liệu tự động từ API (mỗi 2 phút).

💬 Chatbot thông minh hiểu ngôn ngữ tự nhiên tiếng Việt & Anh.

📈 Biểu đồ trực quan về AQI hiện tại và dự báo.

🧠 Dự báo đa thời điểm: +1h, +3h, +6h.

❤️ Khuyến nghị sức khỏe cá nhân hóa dựa trên tuổi và bệnh lý.


4. 🔗 Liên kết GitHub dự án

👉 Dự án đầy đủ: https://github.com/agnessktt/HTTM_NHOM8