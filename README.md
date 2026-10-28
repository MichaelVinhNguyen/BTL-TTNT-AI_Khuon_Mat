# HỆ THỐNG ĐIỂM DANH & NHẬN DIỆN KHUÔN MẶT AI-PRO • 4D CYBER HUD COCKPIT
> **Học viện Công nghệ Bưu chính Viễn thông (PTIT)**  
> **Đồ án môn học: Trí Tuệ Nhân Tạo (Artificial Intelligence)**

---

## 🌟 Giới thiệu
Dự án phát triển hệ thống nhận diện khuôn mặt và điểm danh tự động thông minh với giao diện hiện đại **CustomTkinter Dark Obsidian & 4D Cyber HUD Hologram**. Hệ thống tích hợp song song cả mô hình thị giác máy tính truyền thống (OpenCV) và các thuật toán học máy (Machine Learning) từ thư viện Scikit-Learn.

---

## 🚀 Các Thuật Toán Tích Hợp
1. **k-NN (k-Nearest Neighbors)** - Phân lớp theo khoảng cách Euclide.
2. **Decision Tree (Cây quyết định)** - Tiêu chuẩn tối ưu Gini / Entropy.
3. **Gaussian Naive Bayes** - Phân loại dựa trên xác suất có điều kiện Bayes.
4. **SVM (Support Vector Machine)** - Phân tách siêu phẳng cực đại.
5. **Random Forest Ensemble** - Rừng cây quyết định ngẫu nhiên tăng cường độ chính xác.
6. **MLP Neural Network** - Mạng nơ-ron truyền thẳng đa tầng.
7. **LBPH (Local Binary Patterns Histograms)** - Thuật toán nhận diện cục bộ nguyên bản của OpenCV.

---

## 🛠️ Tính Năng Nổi Bật
- **Giao diện 4D Cyber HUD Cockpit**: Thiết kế Cyberpunk Dark Obsidian hiện đại với CustomTkinter.
- **Mô phỏng lưới điểm sinh trắc học 4D**: Facial Mesh Simulation trực quan theo thời gian thực.
- **Bộ lọc tiền xử lý ảnh thời gian thực**: Hỗ trợ tăng cường chất lượng ảnh trực tiếp khi trích xuất và huấn luyện.
- **Điểm danh tự động & Quét mã QR**: Tích hợp quét mã QR, tự động đối soát thông tin sinh viên/người dùng.
- **Cảnh báo thông minh**: Tự động thông báo và ngăn chặn trùng lặp điểm danh trong khoảng thời gian định trước (1 giờ).
- **Xuất báo cáo đa định dạng**: Tự động xuất và tổng hợp báo cáo điểm danh ra file Excel (`.xlsx`) chuẩn định dạng PTIT và file `.csv`.

---

## 📦 Cài Đặt & Sử Dụng

### 1. Yêu cầu môi trường
- Python 3.8 trở lên
- Webcam hoạt động bình thường

### 2. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng
Có thể khởi chạy nhanh bằng file script:
```bash
run_app.bat
```
Hoặc chạy trực tiếp bằng dòng lệnh:
```bash
python face_recognition_pro.py
```

---

## 📁 Cấu Trúc Thư Mục
```text
├── dataset/                    # Thư mục lưu trữ dữ liệu ảnh khuôn mặt người dùng
├── captured/                   # Thư mục lưu ảnh chụp điểm danh
├── face_recognition_pro.py      # Mã nguồn chính của ứng dụng AI-PRO
├── generate_reports.py         # Module sinh báo cáo điểm danh
├── haarcascade_frontalface_default.xml # Bộ phân loại Haar Cascade phát hiện khuôn mặt
├── requirements.txt            # Danh sách các thư viện phụ thuộc
├── run_app.bat                 # Script chạy nhanh ứng dụng
├── BaoCao BTL Trí Tuệ Nhân Tạo.docx # Báo cáo chi tiết đồ án
└── README.md                   # Tài liệu hướng dẫn dự án
```
