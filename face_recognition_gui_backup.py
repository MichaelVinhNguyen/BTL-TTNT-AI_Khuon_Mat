# -*- coding: utf-8 -*-
import cv2
import os
import datetime
import numpy as np
import pickle
import qrcode
import csv
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from PIL import Image, ImageTk
import queue

# ==========================
# Thư mục lưu dữ liệu
# ==========================
BASE_DIR = "dataset"
CAPTURE_DIR = "captured"
LOG_FILE = "attendance_log.csv"

for d in [BASE_DIR, CAPTURE_DIR]:
    os.makedirs(d, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==========================
# Các mục chức năng xử lý ảnh
# ==========================
NO_FILTER, SHARPEN, BW, EDGE, SEPIA, SMOOTH = range(6)
FILTER_NAMES = {
    NO_FILTER: "Không",
    SHARPEN: "Làm Nét",
    BW: "Đen Trắng",
    EDGE: "Cạnh",
    SEPIA: "Nâu Cổ",
    SMOOTH: "Làm Mịn",
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

# ==========================
# Ghi log điểm danh
# ==========================
def log_attendance(label):
    logged_today = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    logged_today.add(row[0])

    if label not in logged_today:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([label, timestamp()])
        print(f"[LOG] {label} Đã được điểm danh lúc {timestamp()}")
        return True
    return False

# ==========================
# Huấn luyện model
# ==========================
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = {}
    x_train = []
    y_labels = []
    current_id = 0

    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                if label not in labels.values():
                    labels[current_id] = label
                    id_val = current_id
                    current_id += 1
                else:
                    id_val = [k for k, v in labels.items() if v == label][0]

                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                x_train.append(img)
                y_labels.append(id_val)

    if len(x_train) > 0:
        recognizer.train(x_train, np.array(y_labels))
        recognizer.save("trainer.yml")
        with open("labels.pkl", "wb") as f:
            pickle.dump(labels, f)
        return True
    return False

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Nhận Diện Khuôn Mặt AI-Pro")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        
        # Variables
        self.cap = None
        self.current_filter = NO_FILTER
        self.is_capturing = False
        self.is_recognizing = False
        self.saved_faces = []
        self.recognized_users = set()
        self.recognizer = None
        self.labels = {}
        self.current_recognized_person = None
        self.current_confidence = 0
        
        # Queue for thread communication
        self.frame_queue = queue.Queue()
        
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        # Title Frame
        title_frame = tk.Frame(self.root, bg='#1a252f', height=100)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🤖 HỆ THỐNG NHẬN DIỆN KHUÔN MẶT AI-PRO 🤖", 
                               font=('Arial', 26, 'bold'), bg='#1a252f', fg='#00ff88')
        title_label.pack(pady=25)
        
        # Main Container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Camera Control Frame (Top)
        camera_frame = tk.Frame(main_container, bg='#1a252f', height=60)
        camera_frame.pack(fill=tk.X, padx=10, pady=5)
        camera_frame.pack_propagate(False)
        
        camera_label = tk.Label(camera_frame, text="📷 ĐIỀU KHIỂN CAMERA", 
                               font=('Arial', 14, 'bold'), bg='#1a252f', fg='#00ff88')
        camera_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Camera buttons
        self.open_camera_btn = tk.Button(camera_frame, text="📹 MỞ CAMERA", 
                                       command=self.open_camera, bg='#27ae60', fg='white',
                                       font=('Arial', 12, 'bold'), width=15, height=2, relief=tk.RAISED, bd=3)
        self.open_camera_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.close_camera_btn = tk.Button(camera_frame, text="📹 ĐÓNG CAMERA", 
                                        command=self.close_camera, bg='#e74c3c', fg='white',
                                        font=('Arial', 12, 'bold'), width=15, height=2, relief=tk.RAISED, bd=3, state=tk.DISABLED)
        self.close_camera_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Camera status
        self.camera_status_label = tk.Label(camera_frame, text="🔴 Camera: Đã đóng", 
                                          font=('Arial', 12, 'bold'), bg='#1a252f', fg='#e74c3c')
        self.camera_status_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Left Panel - Video Display
        left_panel = tk.Frame(main_container, bg='#34495e', width=650, height=500)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Camera frame with border
        camera_display_frame = tk.Frame(left_panel, bg='#00ff88', relief=tk.RAISED, bd=3)
        camera_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.video_label = tk.Label(camera_display_frame, bg='black', text="📷 CAMERA SẼ HIỂN THỊ Ở ĐÂY", 
                                   font=('Arial', 16, 'bold'), fg='#00ff88')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # Right Panel - Controls
        right_panel = tk.Frame(main_container, bg='#34495e', width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Mode Selection
        mode_frame = tk.LabelFrame(right_panel, text="🎯 CHỌN CHẾ ĐỘ HOẠT ĐỘNG", 
                                  font=('Arial', 14, 'bold'), bg='#34495e', fg='#00ff88')
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.mode_var = tk.StringVar(value="capture")
        tk.Radiobutton(mode_frame, text="� CHỤP ẢNH & ĐĂNG KÝ", variable=self.mode_var, 
                      value="capture", bg='#34495e', fg='white', selectcolor='#2c3e50',
                      font=('Arial', 11, 'bold'), command=self.switch_mode).pack(anchor=tk.W, padx=10, pady=8)
        tk.Radiobutton(mode_frame, text="🔍 NHẬN DIỆN & ĐIỂM DANH", variable=self.mode_var, 
                      value="recognition", bg='#34495e', fg='white', selectcolor='#2c3e50',
                      font=('Arial', 11, 'bold'), command=self.switch_mode).pack(anchor=tk.W, padx=10, pady=8)
        
        # Filter Selection (for capture mode)
        self.filter_frame = tk.LabelFrame(right_panel, text="🎨 BỘ LỌC ẢNH", 
                                         font=('Arial', 14, 'bold'), bg='#34495e', fg='#00ff88')
        self.filter_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.filter_var = tk.StringVar(value="0")
        for i, (key, name) in enumerate(FILTER_NAMES.items()):
            tk.Radiobutton(self.filter_frame, text=f"{i}. {name}", variable=self.filter_var, 
                          value=str(i), bg='#34495e', fg='white', selectcolor='#2c3e50',
                          font=('Arial', 10), command=self.change_filter).pack(anchor=tk.W, padx=10, pady=3)
        
        # User Info Frame (for capture mode)
        self.user_frame = tk.LabelFrame(right_panel, text="👤 THÔNG TIN ĐĂNG KÝ", 
                                       font=('Arial', 14, 'bold'), bg='#34495e', fg='#00ff88')
        self.user_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Name field
        name_container = tk.Frame(self.user_frame, bg='#34495e')
        name_container.pack(fill=tk.X, padx=10, pady=(8, 5))
        tk.Label(name_container, text="📛 Họ Tên:", bg='#34495e', fg='white', 
                font=('Arial', 11, 'bold'), width=12, anchor='w').pack(side=tk.LEFT)
        self.name_entry = tk.Entry(name_container, font=('Arial', 11), bg='#2c3e50', fg='white', 
                                  insertbackground='white', relief=tk.RAISED, bd=2)
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # ID field
        id_container = tk.Frame(self.user_frame, bg='#34495e')
        id_container.pack(fill=tk.X, padx=10, pady=(5, 8))
        tk.Label(id_container, text="🆔 Mã Số:", bg='#34495e', fg='white', 
                font=('Arial', 11, 'bold'), width=12, anchor='w').pack(side=tk.LEFT)
        self.id_entry = tk.Entry(id_container, font=('Arial', 11), bg='#2c3e50', fg='white', 
                                insertbackground='white', relief=tk.RAISED, bd=2)
        self.id_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Save button for registration
        self.save_reg_btn = tk.Button(self.user_frame, text="💾 LƯU ĐĂNG KÝ & TẠO QR", 
                                      command=self.save_registration, bg='#f39c12', fg='white',
                                      font=('Arial', 11, 'bold'), relief=tk.RAISED, bd=3, state=tk.DISABLED)
        self.save_reg_btn.pack(pady=(5, 8))
        
        # Recognition Info Frame (for recognition mode)
        self.recognition_frame = tk.LabelFrame(right_panel, text="🔍 THÔNG TIN NHẬN DIỆN", 
                                              font=('Arial', 14, 'bold'), bg='#34495e', fg='#00ff88')
        
        # Current recognized person
        person_container = tk.Frame(self.recognition_frame, bg='#34495e')
        person_container.pack(fill=tk.X, padx=10, pady=(8, 5))
        tk.Label(person_container, text="👤 Người Dùng:", bg='#34495e', fg='white', 
                font=('Arial', 11, 'bold'), width=12, anchor='w').pack(side=tk.LEFT)
        self.current_person_label = tk.Label(person_container, text="Chưa nhận diện", bg='#2c3e50', fg='#00ff88', 
                                           font=('Arial', 11, 'bold'), relief=tk.SUNKEN, bd=2)
        self.current_person_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Confidence level
        conf_container = tk.Frame(self.recognition_frame, bg='#34495e')
        conf_container.pack(fill=tk.X, padx=10, pady=(5, 8))
        tk.Label(conf_container, text="📊 Độ Tin Cậy:", bg='#34495e', fg='white', 
                font=('Arial', 11, 'bold'), width=12, anchor='w').pack(side=tk.LEFT)
        self.confidence_label = tk.Label(conf_container, text="0%", bg='#2c3e50', fg='#00ff88', 
                                       font=('Arial', 11, 'bold'), relief=tk.SUNKEN, bd=2)
        self.confidence_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Save attendance button
        self.save_attendance_btn = tk.Button(self.recognition_frame, text="🔔 LƯU ĐIỂM DANH", 
                                           command=self.save_attendance, bg='#e74c3c', fg='white',
                                           font=('Arial', 11, 'bold'), relief=tk.RAISED, bd=3, state=tk.DISABLED)
        self.save_attendance_btn.pack(pady=(5, 8))
        
        # Control Buttons
        control_frame = tk.Frame(right_panel, bg='#34495e')
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Row 1 - Main controls
        row1 = tk.Frame(control_frame, bg='#34495e')
        row1.pack(fill=tk.X, pady=(0, 5))
        
        self.start_btn = tk.Button(row1, text="🚀 BẮT ĐẦU", 
                                  command=self.start_camera, bg='#27ae60', fg='white',
                                  font=('Arial', 11, 'bold'), width=13, height=2, relief=tk.RAISED, bd=3)
        self.start_btn.pack(side=tk.LEFT, padx=3)
        
        self.capture_btn = tk.Button(row1, text="📸 CHỤP ẢNH", 
                                    command=self.capture_photo, bg='#3498db', fg='white',
                                    font=('Arial', 11, 'bold'), width=13, height=2, state=tk.DISABLED, relief=tk.RAISED, bd=3)
        self.capture_btn.pack(side=tk.LEFT, padx=3)
        
        self.stop_btn = tk.Button(row1, text="⏹️ DỪNG LẠI", 
                                 command=self.stop_camera, bg='#e74c3c', fg='white',
                                 font=('Arial', 11, 'bold'), width=13, height=2, state=tk.DISABLED, relief=tk.RAISED, bd=3)
        self.stop_btn.pack(side=tk.LEFT, padx=3)
        
        # Row 2 - Additional controls
        row2 = tk.Frame(control_frame, bg='#34495e')
        row2.pack(fill=tk.X, pady=(5, 0))
        
        self.clear_btn = tk.Button(row2, text="🗑️ XÓA DỮ LIỆU", 
                                   command=self.clear_data, bg='#95a5a6', fg='white',
                                   font=('Arial', 10, 'bold'), width=13, height=1, relief=tk.RAISED, bd=2)
        self.clear_btn.pack(side=tk.LEFT, padx=3)
        
        self.view_btn = tk.Button(row2, text="📊 XEM DANH SÁCH", 
                                 command=self.view_records, bg='#8e44ad', fg='white',
                                 font=('Arial', 10, 'bold'), width=13, height=1, relief=tk.RAISED, bd=2)
        self.view_btn.pack(side=tk.LEFT, padx=3)
        
        self.export_btn = tk.Button(row2, text="💾 XUẤT FILE", 
                                   command=self.export_data, bg='#16a085', fg='white',
                                   font=('Arial', 10, 'bold'), width=13, height=1, relief=tk.RAISED, bd=2)
        self.export_btn.pack(side=tk.LEFT, padx=3)
        
        # Status Frame
        status_frame = tk.LabelFrame(right_panel, text="📊 TRẠNG THÁI HỆ THỐNG", 
                                    font=('Arial', 14, 'bold'), bg='#34495e', fg='#00ff88')
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.status_label = tk.Label(status_frame, text="✅ SẴN SÀNG HOẠT ĐỘNG", 
                                    bg='#34495e', fg='#27ae60', font=('Arial', 11, 'bold'))
        self.status_label.pack(padx=10, pady=8)
        
        # Activity Log
        log_frame = tk.LabelFrame(right_panel, text="📋 NHẬT KÝ HOẠT ĐỘNG", 
                                  font=('Arial', 14, 'bold'), bg='#34495e', fg='#00ff88')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollable text widget
        log_scroll = tk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_frame, height=10, width=40, bg='#1a252f', fg='#00ff88', 
                               font=('Courier', 9), yscrollcommand=log_scroll.set)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        log_scroll.config(command=self.log_text.yview)
        
        # Footer
        footer_frame = tk.Frame(self.root, bg='#1a252f', height=40)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        footer_frame.pack_propagate(False)
        
        footer_content = tk.Frame(footer_frame, bg='#1a252f')
        footer_content.pack(expand=True)
        
        footer_label = tk.Label(footer_content, text="🤖 Development by QuangVinh Nguyen c226 All resouse right 🤖", 
                               font=('Arial', 10, 'bold'), bg='#1a252f', fg='#00ff88')
        footer_label.pack(pady=10)
        
    def load_model(self):
        if os.path.exists("trainer.yml") and os.path.exists("labels.pkl"):
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read("trainer.yml")
                with open("labels.pkl", "rb") as f:
                    self.labels = pickle.load(f)
                self.log("Model đã được tải thành công")
            except Exception as e:
                self.log(f"Lỗi tải model: {str(e)}")
        else:
            self.log("Chưa có model, cần đăng ký người dùng trước")
    
    def log(self, message, log_type="INFO"):
        timestamp_str = datetime.datetime.now().strftime("%H:%M:%S")
        icons = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
        icon = icons.get(log_type, "ℹ️")
        
        # Add to GUI log
        self.log_text.insert(tk.END, f"[{timestamp_str}] {icon} {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
        # Also save to daily log file
        daily_log_file = f"daily_log_{datetime.datetime.now().strftime('%Y%m%d')}.txt"
        with open(daily_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp_str}] [{log_type}] {message}\n")
    
    def switch_mode(self):
        mode = self.mode_var.get()
        if mode == "capture":
            self.filter_frame.pack(fill=tk.X, padx=10, pady=10, after=self.user_frame.master.winfo_children()[1])
            self.user_frame.pack(fill=tk.X, padx=10, pady=10)
            self.recognition_frame.pack_forget()
            self.status_label.config(text="� Chế độ chụp ảnh & đăng ký", fg='#3498db')
            self.log("Chuyển sang chế độ: CHỤP ẢNH & ĐĂNG KÝ", "INFO")
        else:
            self.filter_frame.pack_forget()
            self.user_frame.pack_forget()
            self.recognition_frame.pack(fill=tk.X, padx=10, pady=10)
            self.status_label.config(text="🔍 Chế độ nhận diện & điểm danh", fg='#9b59b6')
            self.log("Chuyển sang chế độ: NHẬN DIỆN & ĐIỂM DANH", "INFO")
    
    def change_filter(self):
        self.current_filter = int(self.filter_var.get())
        self.log(f"Đổi bộ lọc: {FILTER_NAMES[self.current_filter]}", "INFO")
    
    def open_camera(self):
        """Mở camera riêng biệt"""
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi Camera", "❌ Không thể mở camera! Vui lòng kiểm tra kết nối.")
                self.log("Không thể mở camera", "ERROR")
                return
            
            # Update UI
            self.open_camera_btn.config(state=tk.DISABLED)
            self.close_camera_btn.config(state=tk.NORMAL)
            self.camera_status_label.config(text="🟢 Camera: Đã mở", fg='#27ae60')
            self.start_btn.config(state=tk.NORMAL)
            
            self.log("📹 Đã mở camera thành công", "SUCCESS")
            
            # Start preview thread
            self.preview_thread = threading.Thread(target=self.camera_preview)
            self.preview_thread.daemon = True
            self.preview_thread.start()
    
    def close_camera(self):
        """Đóng camera riêng biệt"""
        if self.cap:
            self.cap.release()
            self.cap = None
            
            # Stop any running operations
            self.is_capturing = False
            self.is_recognizing = False
            
            # Update UI
            self.open_camera_btn.config(state=tk.NORMAL)
            self.close_camera_btn.config(state=tk.DISABLED)
            self.camera_status_label.config(text="🔴 Camera: Đã đóng", fg='#e74c3c')
            self.start_btn.config(state=tk.DISABLED)
            self.capture_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.save_attendance_btn.config(state=tk.DISABLED)
            
            # Clear video display
            self.video_label.config(image="", text="📷 CAMERA SẼ HIỂN THỊ Ở ĐÂY")
            
            # Clear recognition info
            self.clear_recognition_info()
            
            self.log("📹 Đã đóng camera", "INFO")
    
    def camera_preview(self):
        """Hiển thị preview camera khi mở"""
        while self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Only show preview, no processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img=img)
            
            # Update GUI in main thread
            self.root.after(0, self.update_video_display, photo)
            
            # Small delay to prevent high CPU usage
            cv2.waitKey(30)
    
    def start_camera(self):
        """Bắt đầu chế độ hoạt động (đăng ký hoặc nhận diện)"""
        if not self.cap:
            messagebox.showwarning("Cảnh Báo", "⚠️ Vui lòng mở camera trước!")
            return
        
        mode = self.mode_var.get()
        if mode == "capture":
            self.is_capturing = True
            self.status_label.config(text="📷 Đang chụp ảnh để đăng ký...", fg='#27ae60')
            self.log("Bắt đầu chế độ CHỤP ẢNH & ĐĂNG KÝ", "SUCCESS")
        else:
            self.is_recognizing = True
            self.status_label.config(text="🔍 Đang nhận diện khuôn mặt...", fg='#9b59b6')
            self.log("Bắt đầu chế độ NHẬN DIỆN & ĐIỂM DANH", "SUCCESS")
        
        self.start_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start processing thread
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def update_video(self):
        while self.is_capturing or self.is_recognizing:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if self.is_capturing:
                frame = self.process_capture_frame(frame)
            elif self.is_recognizing:
                frame = self.process_recognition_frame(frame)
            
            # Convert frame for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update GUI in main thread
            self.root.after(0, self.update_video_display, photo)
    
    def process_capture_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        
        frame = apply_filter(frame, self.current_filter)
        
        # Add filter name
        cv2.putText(frame, f"Filter: {FILTER_NAMES[self.current_filter]}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        return frame
    
    def process_recognition_frame(self, frame):
        if not self.recognizer:
            cv2.putText(frame, "❌ No Model", (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_best_person = None
        current_best_confidence = 0
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = self.recognizer.predict(roi_gray)
            
            if conf >= 45 and conf <= 85:
                label = self.labels[id_]
                cv2.putText(frame, f"✅ {label} ({int(conf)}%)", (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                # Keep track of best recognition
                if conf > current_best_confidence:
                    current_best_person = label
                    current_best_confidence = conf
            else:
                cv2.putText(frame, "❌ Unknown", (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)
        
        # Update UI with current recognition info
        if current_best_person:
            self.current_recognized_person = current_best_person
            self.current_confidence = current_best_confidence
            
            # Update UI labels in main thread
            self.root.after(0, self.update_recognition_info, current_best_person, current_best_confidence)
        else:
            self.current_recognized_person = None
            self.current_confidence = 0
            self.root.after(0, self.clear_recognition_info)
        
        return frame
    
    def update_recognition_info(self, person, confidence):
        """Update recognition info in UI"""
        self.current_person_label.config(text=person, fg='#00ff88')
        self.confidence_label.config(text=f"{int(confidence)}%")
        self.save_attendance_btn.config(state=tk.NORMAL)
    
    def clear_recognition_info(self):
        """Clear recognition info in UI"""
        self.current_person_label.config(text="Chưa nhận diện", fg='#95a5a6')
        self.confidence_label.config(text="0%")
        self.save_attendance_btn.config(state=tk.DISABLED)
    
    def update_video_display(self, photo):
        self.video_label.config(image=photo)
        self.video_label.image = photo
    
    def capture_photo(self):
        if self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    for i, (x, y, w, h) in enumerate(faces, start=1):
                        face_img = frame[y:y+h, x:x+w]
                        fname = f"face_{ts}_{i}.jpg"
                        path = os.path.join(CAPTURE_DIR, fname)
                        cv2.imwrite(path, face_img)
                        self.saved_faces.append(path)
                        self.log(f"✅ Đã lưu ảnh: {fname}", "SUCCESS")
                    
                    messagebox.showinfo("Thành Công", f"📸 Đã chụp {len(faces)} khuôn mặt!")
                    self.log(f"Chụp thành công {len(faces)} khuôn mặt", "SUCCESS")
                    
                    # Enable save button if we have photos and user info
                    if self.name_entry.get().strip() and self.id_entry.get().strip():
                        self.save_reg_btn.config(state=tk.NORMAL)
                else:
                    messagebox.showwarning("Cảnh Báo", "⚠️ Không phát hiện khuôn mặt! Vui lòng nhìn thẳng vào camera.")
                    self.log("Không phát hiện khuôn mặt", "WARNING")
    
    def save_attendance(self):
        """Lưu thông tin điểm danh chi tiết khi bấm nút chuông"""
        if not self.current_recognized_person:
            messagebox.showwarning("Cảnh Báo", "⚠️ Chưa nhận diện được người dùng!")
            return
        
        # Kiểm tra đã điểm danh hôm nay chưa
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        attendance_file = f"attendance_detailed_{today}.csv"
        
        # Check if already logged today
        already_logged = False
        if os.path.exists(attendance_file):
            with open(attendance_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2 and row[0] == self.current_recognized_person:
                        already_logged = True
                        break
        
        if already_logged:
            messagebox.showwarning("Thông Báo", f"⚠️ {self.current_recognized_person} đã điểm danh hôm nay rồi!")
            self.log(f"{self.current_recognized_person} đã điểm danh hôm nay", "WARNING")
            return
        
        # Save detailed attendance
        with open(attendance_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if os.stat(attendance_file).st_size == 0:  # File is empty, write header
                writer.writerow(["Người Dùng", "Thời Gian", "Ngày", "Độ Tin Cậy", "Ghi Chú"])
            
            current_time = timestamp()
            writer.writerow([
                self.current_recognized_person,
                current_time,
                today,
                f"{int(self.current_confidence)}%",
                "Điểm danh bằng nhận diện khuôn mặt"
            ])
        
        # Also save to main attendance log
        log_attendance(self.current_recognized_person)
        
        # Show success message
        messagebox.showinfo("Điểm Danh Thành Công", 
                          f"🔔 Đã lưu điểm danh thành công!\n\n"
                          f"👤 Người dùng: {self.current_recognized_person}\n"
                          f"🕐 Thời gian: {current_time}\n"
                          f"📊 Độ tin cậy: {int(self.current_confidence)}%\n"
                          f"📁 File: {attendance_file}")
        
        self.log(f"🔔 {self.current_recognized_person} đã điểm danh thành công (độ tin cậy: {int(self.current_confidence)}%)", "SUCCESS")
        
        # Disable button after successful attendance
        self.save_attendance_btn.config(state=tk.DISABLED)
        self.current_person_label.config(text="Đã điểm danh", fg='#f39c12')
    
    def save_registration(self):
        """Lưu đăng ký người dùng với nút bấm riêng"""
        if not self.saved_faces:
            messagebox.showwarning("Cảnh Báo", "⚠️ Chưa chụp ảnh khuôn mặt! Vui lòng chụp ảnh trước.")
            self.log("Chưa có ảnh để đăng ký", "WARNING")
            return
        
        self.save_user_data()
        self.save_reg_btn.config(state=tk.DISABLED)
    
    def clear_data(self):
        """Xóa dữ liệu người dùng tạm thời"""
        result = messagebox.askyesno("Xác Nhận", "🗑️ Bạn có chắc muốn xóa dữ liệu tạm thời?")
        if result:
            self.saved_faces.clear()
            self.name_entry.delete(0, tk.END)
            self.id_entry.delete(0, tk.END)
            self.save_reg_btn.config(state=tk.DISABLED)
            self.log("Đã xóa dữ liệu tạm thời", "INFO")
            messagebox.showinfo("Thành Công", "✅ Đã xóa dữ liệu tạm thời!")
    
    def view_records(self):
        """Xem danh sách người dùng và điểm danh"""
        view_window = tk.Toplevel(self.root)
        view_window.title("📊 Danh Sách Dữ Liệu")
        view_window.geometry("800x600")
        view_window.configure(bg='#2c3e50')
        
        # Notebook for tabs
        notebook = ttk.Notebook(view_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Users tab
        users_frame = tk.Frame(notebook, bg='#34495e')
        notebook.add(users_frame, text="👥 Người Dùng Đăng Ký")
        
        users_text = tk.Text(users_frame, bg='#2c3e50', fg='white', font=('Courier', 10))
        users_scroll = tk.Scrollbar(users_frame, command=users_text.yview)
        users_text.configure(yscrollcommand=users_scroll.set)
        
        users_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        users_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Load user registrations
        if os.path.exists("user_registrations.csv"):
            with open("user_registrations.csv", "r", encoding="utf-8") as f:
                users_text.insert(tk.END, f"{'ID':<25} {'Họ Tên':<20} {'Mã Số':<15} {'Thời Gian':<20}\n")
                users_text.insert(tk.END, "="*85 + "\n")
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 4:
                        users_text.insert(tk.END, f"{row[0]:<25} {row[1]:<20} {row[2]:<15} {row[3]:<20}\n")
        else:
            users_text.insert(tk.END, "Chưa có dữ liệu người dùng đăng ký.")
        
        # Attendance tab
        attendance_frame = tk.Frame(notebook, bg='#34495e')
        notebook.add(attendance_frame, text="📝 Lịch Sử Điểm Danh")
        
        attendance_text = tk.Text(attendance_frame, bg='#2c3e50', fg='white', font=('Courier', 10))
        attendance_scroll = tk.Scrollbar(attendance_frame, command=attendance_text.yview)
        attendance_text.configure(yscrollcommand=attendance_scroll.set)
        
        attendance_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        attendance_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Load attendance records
        if os.path.exists("attendance_log.csv"):
            with open("attendance_log.csv", "r", encoding="utf-8") as f:
                attendance_text.insert(tk.END, f"{'Người Dùng':<25} {'Thời Gian':<20}\n")
                attendance_text.insert(tk.END, "="*50 + "\n")
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        attendance_text.insert(tk.END, f"{row[0]:<25} {row[1]:<20}\n")
        else:
            attendance_text.insert(tk.END, "Chưa có dữ liệu điểm danh.")
    
    def export_data(self):
        """Xuất dữ liệu ra file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Chọn nơi lưu file xuất dữ liệu"
        )
        
        if file_path:
            try:
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["BÁO CÁO HỆ THỐNG NHẬN DIỆN KHUÔN MẶT"])
                    writer.writerow(["Ngày xuất:", timestamp()])
                    writer.writerow([])
                    
                    # User registrations
                    if os.path.exists("user_registrations.csv"):
                        writer.writerow(["DANH SÁCH NGƯỜI DÙNG ĐĂNG KÝ"])
                        with open("user_registrations.csv", "r", encoding="utf-8") as reg_file:
                            reader = csv.reader(reg_file)
                            for row in reader:
                                writer.writerow(row)
                        writer.writerow([])
                    
                    # Attendance records
                    if os.path.exists("attendance_log.csv"):
                        writer.writerow(["LỊCH SỬ ĐIỂM DANH"])
                        with open("attendance_log.csv", "r", encoding="utf-8") as att_file:
                            reader = csv.reader(att_file)
                            for row in reader:
                                writer.writerow(row)
                
                self.log(f"✅ Đã xuất dữ liệu ra: {file_path}", "SUCCESS")
                messagebox.showinfo("Thành Công", f"✅ Đã xuất dữ liệu thành công!\n📁 {file_path}")
            except Exception as e:
                self.log(f"❌ Lỗi xuất dữ liệu: {str(e)}", "ERROR")
                messagebox.showerror("Lỗi", f"❌ Không thể xuất dữ liệu: {str(e)}")
    
    def stop_camera(self):
        """Dừng chế độ hoạt động nhưng không đóng camera"""
        if self.is_capturing:
            self.is_capturing = False
            self.log("Dừng chế độ đăng ký", "INFO")
        elif self.is_recognizing:
            self.is_recognizing = False
            self.recognized_users.clear()
            self.log("Dừng chế độ nhận diện", "INFO")
        
        # Update UI buttons
        self.start_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="✅ Sẵn sàng hoạt động", fg='#27ae60')
        
        # Clear recognition info
        self.clear_recognition_info()
        
        # Return to preview mode
        if self.cap:
            self.preview_thread = threading.Thread(target=self.camera_preview)
            self.preview_thread.daemon = True
            self.preview_thread.start()
    
    def save_user_data(self):
        name = self.name_entry.get().strip()
        mssv = self.id_entry.get().strip()
        
        if not name or not mssv:
            messagebox.showwarning("Thiếu Thông Tin", "⚠️ Vui lòng nhập đầy đủ Họ tên và Mã số!")
            self.log("Thiếu thông tin người dùng", "WARNING")
            return
        
        user_id = f"{name}-{mssv}"
        user_dir = os.path.join(BASE_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        for i, fpath in enumerate(self.saved_faces):
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(user_dir, f"{i}.jpg"), img)
        
        # Generate QR code
        qr = qrcode.make(user_id)
        qr.save(os.path.join(user_dir, "qr.png"))
        
        # Train model
        if train_model():
            self.load_model()
            self.log(f"✅ Đã đăng ký thành công: {user_id}", "SUCCESS")
            messagebox.showinfo("Thành Công", f"🎉 Đã đăng ký thành công người dùng:\n{name} - {mssv}")
            
            # Save registration info
            with open("user_registrations.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([user_id, name, mssv, timestamp()])
        else:
            self.log("❌ Lỗi khi train model", "ERROR")
            messagebox.showerror("Lỗi", "❌ Không thể train model!")
        
        # Clear data
        self.saved_faces.clear()
        self.name_entry.delete(0, tk.END)
        self.id_entry.delete(0, tk.END)
        
        # Clean up captured files
        for fpath in self.saved_faces:
            if os.path.exists(fpath):
                os.remove(fpath)
        self.saved_faces.clear()
    
    def on_closing(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
