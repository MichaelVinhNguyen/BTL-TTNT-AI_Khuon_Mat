# -*- coding: utf-8 -*-
"""
FACE RECOGNITION FULL (CLI Version)
Tích hợp các thuật toán Machine Learning:
  1. k-NN (k-Nearest Neighbors) [Slide 2b]
  2. Decision Tree (Cây quyết định) [Slide 2a]
  3. Naive Bayes (Gaussian Bayes) [Slide 2c]
  4. SVM (Support Vector Machine) [Bổ sung 1]
  5. Random Forest (Rừng ngẫu nhiên) [Bổ sung 2]
  6. MLP Neural Network (Mạng nơ-ron) [Bổ sung 3]
  7. LBPH (OpenCV mặc định)
"""

import cv2
import os
import datetime
import numpy as np
import pickle
import qrcode
import csv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ==========================
# Thư mục lưu dữ liệu
# ==========================
BASE_DIR = "dataset"
CAPTURE_DIR = "captured"
LOG_FILE = "attendance_log.csv"
TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.pkl"
ML_MODELS_FILE = "ml_models.pkl"
IMG_SIZE = (50, 50)

for d in [BASE_DIR, CAPTURE_DIR]:
    os.makedirs(d, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==========================
# Các bộ lọc xử lý ảnh
# ==========================
NO_FILTER, SHARPEN, BW, EDGE, SEPIA, SMOOTH = range(6)
FILTER_NAMES = {
    NO_FILTER: "None",
    SHARPEN: "Sharpen",
    BW: "B/W",
    EDGE: "Edges",
    SEPIA: "Sepia",
    SMOOTH: "Smooth",
}

def apply_filter(img, mode):
    if mode == NO_FILTER:
        return img
    if mode == SHARPEN:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype='float32')
        return cv2.filter2D(img, -1, kernel)
    if mode == BW:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    if mode == EDGE:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if mode == SEPIA:
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]], dtype='float32')
        sepia = cv2.transform(img, kernel)
        return np.clip(sepia, 0, 255).astype('uint8')
    if mode == SMOOTH:
        return cv2.bilateralFilter(img, 9, 75, 75)
    return img

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def extract_face_features(gray_img):
    if gray_img is None or gray_img.size == 0:
        return np.zeros(IMG_SIZE[0] * IMG_SIZE[1], dtype=np.float32)
    resized = cv2.resize(gray_img, IMG_SIZE)
    return (resized.flatten() / 255.0).astype(np.float32)

# ==========================
# Ghi log điểm danh
# ==========================
def log_attendance(label, algorithm="LBPH"):
    logged_today = set()
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[1].startswith(today_str):
                    logged_today.add(row[0])

    if label not in logged_today:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([label, timestamp(), algorithm])
        print(f"[LOG] {label} Đã được điểm danh lúc {timestamp()} [{algorithm}]")
        return True
    return False

# ==========================
# Huấn luyện model
# ==========================
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = {}
    x_train_raw = []
    x_train_feat = []
    y_labels = []
    current_id = 0

    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')) and file != "qr.png":
                path = os.path.join(root, file)
                label = os.path.basename(root)
                if label not in labels.values():
                    labels[current_id] = label
                    id_val = current_id
                    current_id += 1
                else:
                    id_val = [k for k, v in labels.items() if v == label][0]

                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    x_train_raw.append(img)
                    x_train_feat.append(extract_face_features(img))
                    y_labels.append(id_val)

    if len(x_train_raw) > 0:
        # 1. OpenCV LBPH
        try:
            recognizer.train(x_train_raw, np.array(y_labels))
            recognizer.save(TRAINER_FILE)
        except Exception as e:
            print(f"[WARN] Lỗi LBPH: {e}")

        with open(LABELS_FILE, "wb") as f:
            pickle.dump(labels, f)

        # 2. Machine Learning models
        X_feat = np.array(x_train_feat, dtype=np.float32)
        y_arr = np.array(y_labels, dtype=np.int32)

        # Mẫu âm tính giả định (Unknown, nhãn -1)
        np.random.seed(42)
        dim = IMG_SIZE[0] * IMG_SIZE[1]
        n_neg = max(len(X_feat), 8)
        neg_feat = np.random.uniform(0.0, 1.0, (n_neg, dim)).astype(np.float32)
        X_all = np.vstack([X_feat, neg_feat])
        y_all = np.concatenate([y_arr, np.array([-1] * n_neg, dtype=np.int32)])

        ml_models = {
            "knn": KNeighborsClassifier(n_neighbors=1, algorithm='brute', n_jobs=1),
            "decision_tree": DecisionTreeClassifier(max_depth=12, random_state=42),
            "naive_bayes": GaussianNB(),
            "svm": SVC(kernel='linear', probability=True, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=1),
            "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
        }

        trained = {}
        for k, m in ml_models.items():
            try:
                m.fit(X_all, y_all)
                trained[k] = m
            except Exception as e:
                print(f"[WARN] Lỗi {k}: {e}")

        with open(ML_MODELS_FILE, "wb") as f:
            pickle.dump(trained, f)

        print("[INFO] Đã train thành công 7 thuật toán: k-NN, Decision Tree, Naive Bayes, SVM, Random Forest, MLP, LBPH.")
    else:
        print("[WARN] Không có dữ liệu để train.")

# ==========================
# Chế độ chụp ảnh
# ==========================
def capture_mode():
    cap = cv2.VideoCapture(0)
    current_filter = NO_FILTER
    saved_faces = []

    window_title = "Capture Mode | (0-5)Filter | S=Save Q=Quit"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 960, 540)

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        display = apply_filter(frame, current_filter)
        cv2.putText(display, f"Filter: {FILTER_NAMES[current_filter]}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow(window_title, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in [ord(str(i)) for i in range(6)]:
            current_filter = int(chr(key))
        elif key == ord('s'):
            if len(faces) > 0:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                for i, (x, y, w, h) in enumerate(faces, start=1):
                    face_img = frame[y:y+h, x:x+w]
                    fname = f"face_{ts}_{i}.jpg"
                    path = os.path.join(CAPTURE_DIR, fname)
                    cv2.imwrite(path, face_img)
                    saved_faces.append(path)
                    print(f"[INFO] Saved {fname}")
            else:
                print("[WARN] Không thấy khuôn mặt!")

    cap.release()
    cv2.destroyAllWindows()

    if saved_faces:
        name = input("Nhập Họ Tên: ").strip()
        mssv = input("Nhập MSSV: ").strip()
        if not name or not mssv:
            print("[ERROR] Tên hoặc MSSV rỗng, hủy bỏ lưu!")
            return

        user_id = f"{name}-{mssv}"
        user_dir = os.path.join(BASE_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)

        for i, fpath in enumerate(saved_faces):
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(user_dir, f"{i}.jpg"), img)

        try:
            qr = qrcode.make(user_id)
            qr.save(os.path.join(user_dir, "qr.png"))
        except Exception:
            pass

        train_model()
        print(f"[SUCCESS] Đã lưu thông tin cho: {user_id}")

# ==========================
# Chế độ nhận diện
# ==========================
def recognition_mode():
    if not os.path.exists(LABELS_FILE):
        print("[ERROR] Chưa có dữ liệu, hãy đăng ký trước!")
        return

    with open(LABELS_FILE, "rb") as f:
        labels = pickle.load(f)

    recognizer = None
    if os.path.exists(TRAINER_FILE):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_FILE)

    ml_models = {}
    if os.path.exists(ML_MODELS_FILE):
        with open(ML_MODELS_FILE, "rb") as f:
            ml_models = pickle.load(f)

    algo_list = ["knn", "decision_tree", "naive_bayes", "svm", "random_forest", "mlp", "lbph"]
    algo_names = {
        "knn": "k-NN [Slide 2b]",
        "decision_tree": "Decision Tree [Slide 2a]",
        "naive_bayes": "Naive Bayes [Slide 2c]",
        "svm": "SVM [Bổ sung 1]",
        "random_forest": "Random Forest [Bổ sung 2]",
        "mlp": "MLP Neural [Bổ sung 3]",
        "lbph": "LBPH (OpenCV)"
    }

    current_algo_idx = 0

    window_title = "Recognition Mode | Key 1-7: Switch Algorithm | Q=Quit"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 960, 540)

    cap = cv2.VideoCapture(0)
    recognized_users = set()

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        algo_key = algo_list[current_algo_idx]
        algo_name = algo_names[algo_key]

        cv2.putText(frame, f"Algorithm (Key 1-7): {algo_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label = "Unknown"
            conf_str = ""

            if algo_key == "lbph" and recognizer:
                try:
                    id_, conf = recognizer.predict(roi_gray)
                    if conf <= 85:
                        label = labels.get(id_, "Unknown")
                        conf_val = max(0, min(100, int(100 - (conf * 0.7))))
                        conf_str = f" ({conf_val}%)"
                except Exception:
                    pass
            elif algo_key in ml_models:
                try:
                    model = ml_models[algo_key]
                    feat = extract_face_features(roi_gray).reshape(1, -1)
                    pred_id = model.predict(feat)[0]
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(feat)[0]
                        classes = model.classes_
                        pred_idx = np.where(classes == pred_id)[0][0]
                        conf_val = int(proba[pred_idx] * 100)
                    else:
                        conf_val = 85

                    if pred_id != -1 and conf_val >= 45:
                        label = labels.get(pred_id, "Unknown")
                        conf_str = f" ({conf_val}%)"
                except Exception:
                    pass

            if label != "Unknown":
                cv2.putText(frame, f"[{algo_key.upper()}] {label}{conf_str}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if label not in recognized_users:
                    log_attendance(label, algo_name)
                    recognized_users.add(label)
            else:
                cv2.putText(frame, f"[{algo_key.upper()}] Unknown", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow(window_title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord(str(i)) for i in range(1, 8)]:
            current_algo_idx = int(chr(key)) - 1
            print(f"[SWITCH] Đã chuyển thuật toán sang: {algo_names[algo_list[current_algo_idx]]}")

    cap.release()
    cv2.destroyAllWindows()

# ==========================
# Menu chính
# ==========================
def main():
    while True:
        print("\n===== HỆ THỐNG AI NHẬN DIỆN KHUÔN MẶT PTIT =====")
        print("1. Chụp ảnh & Đăng ký (Capture & Register)")
        print("2. Chế độ nhận diện (Recognition Mode - 7 Thuật toán AI)")
        print("3. Thoát")
        choice = input("Chọn (1/2/3): ").strip()
        if choice == "1":
            capture_mode()
        elif choice == "2":
            recognition_mode()
        elif choice == "3":
            break
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()