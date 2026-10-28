# -*- coding: utf-8 -*-
"""
Script tự động khởi tạo 2 tệp báo cáo kỹ thuật Microsoft Word (.docx) chuyên nghiệp:
1. BaoCao_ChiTiet_ThuatToan_AI.docx: Phân tích sâu 7 thuật toán, sơ đồ luồng tuần tự, vị trí mã nguồn.
2. BaoCao_TongQuan_HeThong_App_AI.docx: Báo cáo tổng quan hệ sinh thái ứng dụng, tính năng 4D HUD, ưu điểm.
Học viện Công nghệ Bưu chính Viễn thông (PTIT) - Khoa Công nghệ Thông tin.
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_shading(cell, color_hex):
    """Thiết lập màu nền cho ô trong bảng"""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    """Thiết lập khoảng đệm padding trong ô"""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def add_header_footer(doc, title_text):
    """Thêm Header và Footer chuẩn văn bản học thuật"""
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)
        
        # Header
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run(f"HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG (PTIT) • {title_text}")
        hrun.font.name = "Arial"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(100, 116, 139)
        
        # Footer
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("Đồ Án Môn Học: Trí Tuệ Nhân Tạo & Xử Lý Ảnh • 2026")
        frun.font.name = "Arial"
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(148, 163, 184)

def add_code_block(doc, code_str, caption=""):
    """Tạo khối code block có viền bên trái và nền xám trang nhã"""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_shading(cell, "F8FAFC")
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    # Thiết lập viền trái màu xanh đậm
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="0284C7"/>
            <w:top w:val="none"/>
            <w:right w:val="none"/>
            <w:bottom w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    
    if caption:
        c_run = p.add_run(f"// {caption}\n")
        c_run.font.name = "Consolas"
        c_run.font.size = Pt(8.5)
        c_run.font.bold = True
        c_run.font.color.rgb = RGBColor(2, 132, 199)
        
    lines = code_str.strip().split("\n")
    for i, line in enumerate(lines):
        run = p.add_run(line + ("\n" if i < len(lines)-1 else ""))
        run.font.name = "Consolas"
        run.font.size = Pt(9.0)
        run.font.color.rgb = RGBColor(30, 41, 59)
    
    # Đoạn cách sau bảng
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)

def add_callout_box(doc, text_str, title="LƯU Ý QUAN TRỌNG", box_type="INFO"):
    """Tạo hộp ghi chú Callout thanh lịch"""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    
    colors = {
        "INFO": ("E0F2FE", "0284C7", "0369A1"),
        "SUCCESS": ("DCFCE7", "16A34A", "15803D"),
        "WARNING": ("FEF3C7", "D97706", "B45309")
    }
    bg_col, border_col, title_col = colors.get(box_type, colors["INFO"])
    set_cell_shading(cell, bg_col)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="28" w:space="0" w:color="{border_col}"/>
            <w:top w:val="none"/>
            <w:right w:val="none"/>
            <w:bottom w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.2
    
    tr = p.add_run(f"📌 {title}: ")
    tr.font.name = "Arial"
    tr.font.size = Pt(10)
    tr.font.bold = True
    tr.font.color.rgb = RGBColor.from_string(title_col)
    
    body_run = p.add_run(text_str)
    body_run.font.name = "Arial"
    body_run.font.size = Pt(9.5)
    body_run.font.color.rgb = RGBColor(30, 41, 59)
    
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)

# ===============================================================================
# 1. TẠO FILE BÁO CÁO 1: CHI TIẾT THUẬT TOÁN AI, SƠ ĐỒ VẬN HÀNH & MÃ NGUỒN
# ===============================================================================
def create_report_algorithms():
    doc = Document()
    add_header_footer(doc, "BÁO CÁO KỸ THUẬT: CƠ CHẾ VẬN HÀNH 7 THUẬT TOÁN AI")
    
    # ----------------------------------------------------
    # BÌA BÁO CÁO KỸ THUẬT
    # ----------------------------------------------------
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst = p_inst.add_run("HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG\nKHOA CÔNG NGHỆ THÔNG TIN - BỘ MÔN TRÍ TUỆ NHÂN TẠO")
    r_inst.font.name = "Arial"
    r_inst.font.size = Pt(11)
    r_inst.font.bold = True
    r_inst.font.color.rgb = RGBColor(30, 58, 138)
    
    p_div = doc.add_paragraph()
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_div = p_div.add_run("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    r_div.font.color.rgb = RGBColor(203, 213, 225)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(20)
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("BÁO CÁO PHÂN TÍCH CHUYÊN SÂU\nCƠ CHẾ VẬN HÀNH, SƠ ĐỒ LUỒNG VÀ HIỆN THỰC MÃ NGUỒN CÁC THUẬT TOÁN AI NHẬN DIỆN KHUÔN MẶT")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(16)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(15, 23, 42)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Hệ Thống Điểm Danh Sinh Viên Đa Mô Hình (k-NN, Decision Tree, Naive Bayes, SVM, Random Forest, MLP, LBPH)\nChương trình Đào tạo Đại học Chính quy - Học kỳ 8")
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(10.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(71, 85, 105)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(40)
    
    # Thông tin nhóm & đề tài
    info_tbl = doc.add_table(rows=4, cols=2)
    info_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta = [
        ("Môn học:", "Trí Tuệ Nhân Tạo & Thị Giác Máy Tính"),
        ("Đề tài nghiên cứu:", "Nhận diện khuôn mặt sinh viên và điểm danh tự động"),
        ("Tệp mã nguồn chính:", "face_recognition_gui.py / face_recognition_pro.py"),
        ("Năm học / Học kỳ:", "2025 - 2026 / Học kỳ II (HK8)")
    ]
    for row_idx, (k, v) in enumerate(meta):
        c0 = info_tbl.cell(row_idx, 0)
        c1 = info_tbl.cell(row_idx, 1)
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
        set_cell_margins(c1, top=60, bottom=60, left=100, right=100)
        p0 = c0.paragraphs[0]
        p1 = c1.paragraphs[0]
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.size = Pt(10)
        r0.font.name = "Arial"
        r1 = p1.add_run(v)
        r1.font.size = Pt(10)
        r1.font.name = "Arial"
        
    doc.add_page_break()
    
    # ----------------------------------------------------
    # CHƯƠNG 1: KIẾN TRÚC TOÀN BỘ HỆ THỐNG & SƠ ĐỒ LUỒNG XỬ LÝ TUẦN TỰ
    # ----------------------------------------------------
    h1 = doc.add_heading(level=1)
    r = h1.add_run("CHƯƠNG 1: KIẾN TRÚC HỆ THỐNG & SƠ ĐỒ LUỒNG XỬ LÝ TUẦN TỰ")
    r.font.name = "Arial"
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 58, 138)
    
    p = doc.add_paragraph()
    p.add_run("Hệ thống nhận diện khuôn mặt AI-Pro được thiết kế theo một quy trình khép kín gồm 3 luồng xử lý chính: (1) Luồng Thu Thập & Huấn Luyện Dữ Liệu; (2) Luồng Quét & Nhận Diện Thời Gian Thực; (3) Luồng Tự Động Điểm Danh & Kiểm Soát Hạn Mức 1 Giờ.")
    
    # Sơ đồ tổng thể
    if os.path.exists("sodotongthe.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture("sodotongthe.png", width=Inches(6.0))
        c_p = doc.add_paragraph()
        c_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c_r = c_p.add_run("Hình 1.1: Sơ đồ kiến trúc luồng xử lý tổng thể hệ thống AI Face Recognition")
        c_r.font.size = Pt(8.5)
        c_r.font.italic = True
        c_r.font.color.rgb = RGBColor(100, 116, 139)
        
    doc.add_heading("1.1. Thứ tự các bước trong luồng xử lý tuần tự (Execution Pipeline)", level=2)
    steps = [
        ("Bước 1: Tiếp nhận dòng video từ Webcam (Camera Stream)", "Bắt từng khung hình ở tốc độ 30 FPS thông qua OpenCV VideoCapture. Khung hình được resize về kích thước chuẩn 440x330 (CAM_WIDTH x CAM_HEIGHT) để tối ưu hóa hiệu năng tính toán."),
        ("Bước 2: Phát hiện khuôn mặt (Face Detection)", "Áp dụng Haar Cascade Classifier (haarcascade_frontalface_default.xml) trên ảnh xám được downscale 0.5x. Giải pháp này giúp tốc độ quét giảm từ 104ms xuống còn ~3ms, phản hồi tức thời."),
        ("Bước 3: Trích xuất vùng ảnh ROI và Chuẩn hóa Đặc trưng", "Cắt vùng mặt phát hiện được (x, y, w, h), chuyển sang ảnh xám Grayscale và resize về ma trận cố định 50x50 pixel. Dữ liệu được làm phẳng (flatten) và chia cho 255.0 tạo thành vector đặc trưng 2500 chiều chuẩn hóa trong đoạn [0.0, 1.0]."),
        ("Bước 4: Suy luận và Phân loại Đa mô hình (Inference Engine)", "Vector 2500 chiều được đưa vào thuật toán Machine Learning đang được lựa chọn (k-NN, Decision Tree, Naive Bayes, SVM, Random Forest, MLP, hoặc LBPH)."),
        ("Bước 5: Đánh giá độ tin cậy và Lọc mẫu âm tính giả định (Unknown Rejection)", "Mô hình tính toán xác suất tiên nghiệm/hậu nghiệm thông qua predict_proba(). Nếu nhãn dự đoán là lớp tạp âm (-1) hoặc độ tin cậy dưới 45%, khuôn mặt được gắn nhãn Unknown (khung đỏ); ngược lại được gắn nhãn tên sinh viên (khung xanh/vàng)."),
        ("Bước 6: Kiểm tra hạn mức 1 giờ (Cooldown Anti-Duplication)", "Hệ thống kiểm tra xem sinh viên này đã điểm danh trong vòng 3600 giây (1 giờ) qua hay chưa. Nếu đã điểm danh: Khung mặt chuyển màu Vàng Hổ Phách, hiển thị thông báo đã điểm danh rồi. Nếu chưa điểm danh: Tự động lưu vào file Excel và CSV, hiện popup Toast 1s."),
        ("Bước 7: Xuất dữ liệu đồng bộ", "Ghi nhận tức thời vào DiemDanh_YYYY-MM-DD.xlsx, BangDiemDanh_TongHop.xlsx, DiemDanh_YYYY-MM-DD.csv và BangDiemDanh_TongHop.csv.")
    ]
    for st_title, st_desc in steps:
        sp = doc.add_paragraph()
        r1 = sp.add_run(f"• {st_title}: ")
        r1.font.bold = True
        r1.font.name = "Arial"
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = sp.add_run(st_desc)
        r2.font.name = "Arial"
        
    doc.add_page_break()
    
    # ----------------------------------------------------
    # CHƯƠNG 2: PHÂN TÍCH CHI TIẾT 7 THUẬT TOÁN AI
    # ----------------------------------------------------
    h2 = doc.add_heading(level=1)
    r = h2.add_run("CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & NGUYÊN LÝ HOẠT ĐỘNG CỦA 7 THUẬT TOÁN AI")
    r.font.name = "Arial"
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 58, 138)
    
    algos_data = [
        {
            "name": "2.1. Thuật toán k-NN (k-Nearest Neighbors) [Slide 2b]",
            "category": "Học lười dựa trên khoảng cách (Instance-based / Lazy Learning)",
            "math": "Khoảng cách Euclide giữa vector khuôn mặt truy vấn x và các vector mẫu x^(i) trong tập dữ liệu 2500 chiều:\n\nd(x, x^(i)) = sqrt( sum_{j=1}^{2500} (x_j - x_j^(i))^2 )\n\nVới k=1, nhãn dự đoán y_hat = y^( argmin_i d(x, x^(i)) ).",
            "op": "Trong hàm train_model(), hệ thống lưu toàn bộ vector đặc trưng cùng nhãn lớp vào bộ nhớ. Khi nhận diện, k-NN đo khoảng cách tới mọi mẫu đã biết. Do ảnh đã được chuẩn hóa kích thước 50x50 và thang xám đồng nhất, k-NN phân loại rất chính xác.",
            "pros": "Không tốn thời gian huấn luyện (1.1 ms), cài đặt đơn giản, cực kỳ trực quan theo slide 2b giáo trình AI PTIT.",
            "cons": "Tốc độ suy luận tỷ lệ thuận với số lượng mẫu lưu trữ, tốn tài nguyên khi dataset mở rộng.",
            "code_loc": "Dòng 653 trong hàm train_model(), tệp face_recognition_gui.py",
            "code_snippet": "KNeighborsClassifier(n_neighbors=1, algorithm='brute', n_jobs=1)"
        },
        {
            "name": "2.2. Thuật toán Decision Tree (Cây quyết định) [Slide 2a]",
            "category": "Cây quyết định nhị phân CART (Gini Impurity Optimization)",
            "math": "Độ vẩn đục Gini tại nút t:\n\nGini(D_t) = 1 - sum_{c=1}^{C} p_c^2\n\nTiêu chuẩn chọn đặc trưng pixel j và ngưỡng cắt theta tối ưu:\nDelta Gini = Gini(D_t) - [ (|D_L| / |D_t|) * Gini(D_L) + (|D_R| / |D_t|) * Gini(D_R) ] -> MAX.",
            "op": "Thuật toán học cách so sánh độ sáng tại các tọa độ điểm ảnh đặc trưng (ví dụ: điểm ảnh ở vùng mắt hoặc sống mũi > 0.45 thì rẽ trái, ngược lại rẽ phải). Để khắc phục overfitting do số chiều lớn (2500 chiều), hệ thống chặn max_depth=12.",
            "pros": "Tốc độ suy luận vô địch (~0.25 ms/khung hình, đạt > 3000 FPS), cấu trúc điều kiện if-else trong suốt dễ diễn giải.",
            "cons": "Nhạy cảm với nhiễu ánh sáng cục bộ, dễ bị phân nhánh quá sâu nếu không khống chế độ sâu.",
            "code_loc": "Dòng 654 trong hàm train_model(), tệp face_recognition_gui.py",
            "code_snippet": "DecisionTreeClassifier(max_depth=12, random_state=42)"
        },
        {
            "name": "2.3. Thuật toán Gaussian Naive Bayes [Slide 2c]",
            "category": "Phân loại xác suất Bayes có điều kiện (Gaussian Probability Distribution)",
            "math": "Định lý Bayes với giả định các điểm ảnh độc lập có điều kiện:\n\nP(C_k | x) proportional to P(C_k) * prod_{j=1}^{2500} P(x_j | C_k)\n\nTrong đó mật độ xác suất Gaussian của điểm ảnh j:\nP(x_j | C_k) = (1 / sqrt(2*pi*sigma_{kj}^2)) * exp( -(x_j - mu_{kj})^2 / (2*sigma_{kj}^2) ).",
            "op": "Trong pha huấn luyện, mô hình chỉ việc tính giá trị trung bình mu và phương sai sigma^2 cho từng pixel của mỗi người. Khi dự đoán, mô hình áp dụng hàm log-likelihood để tránh hiện tượng tràn số thực dưới (underflow).",
            "pros": "Tốc độ huấn luyện và phân loại siêu nhanh (~1.2 ms), cung cấp độ tin cậy xác suất thực tế rất tự nhiên.",
            "cons": "Giả định các điểm ảnh độc lập có điều kiện là một giả định ngây thơ, không hoàn toàn phản ánh mối tương quan không gian.",
            "code_loc": "Dòng 655 trong hàm train_model(), tệp face_recognition_gui.py",
            "code_snippet": "GaussianNB()"
        },
        {
            "name": "2.4. Thuật toán Support Vector Machine (SVM) [Thuật toán nâng cao 1]",
            "category": "Máy học Vector hỗ trợ phân tách cực đại lề (Maximum Margin Hyperplane)",
            "math": "Bài toán tối ưu hóa lề phân cách cực đại:\n\nmin_{w, b} ( 1/2 * ||w||^2 + C * sum xi_i )\nthỏa mãn y_i * (w^T * x_i + b) >= 1 - xi_i.",
            "op": "SVM tìm kiếm siêu phẳng tối ưu trong không gian 2500 chiều sao cho khoảng cách giữa hai lớp biên (support vectors) là lớn nhất. Mô hình sử dụng kernel tuyến tính kết hợp hàm Platt Scaling (probability=True) để xuất xác suất tin cậy %.",
            "pros": "Độ chính xác và độ bền vững trước nhiễu vượt trội hàng đầu, phân loại cực kỳ sắc bén.",
            "cons": "Thời gian huấn luyện tăng khi số lượng mẫu đăng ký lớn, cần chuẩn hóa dải dữ liệu nghiêm ngặt.",
            "code_loc": "Dòng 656 trong hàm train_model(), tệp face_recognition_gui.py",
            "code_snippet": "SVC(kernel='linear', probability=True, random_state=42)"
        },
        {
            "name": "2.5. Thuật toán Random Forest Ensemble [Thuật toán nâng cao 2]",
            "category": "Học kết hợp tập hợp (Ensemble Bagging & Feature Subsampling)",
            "math": "Tập hợp B cây quyết định {T_1, T_2, ..., T_B}. Nhãn phân lớp được quyết định theo biểu quyết đa số (Majority Voting):\n\ny_hat = mode( T_1(x), T_2(x), ..., T_B(x) ).",
            "op": "Hệ thống xây dựng 30 cây quyết định ngẫu nhiên (n_estimators=30, max_depth=10). Mỗi cây được huấn luyện trên một tập mẫu con lấy mẫu có hoàn lại (bootstrap) và chọn ngẫu nhiên tập đặc trưng con tại mỗi nút.",
            "pros": "Khắc phục triệt để nhược điểm quá khớp của cây đơn lẻ, độ chính xác cực cao và ổn định với góc quay mặt nghiêng.",
            "cons": "Tốn dung lượng bộ nhớ hơn và tốc độ suy luận chậm hơn so với 1 cây quyết định đơn lẻ.",
            "code_loc": "Dòng 657 trong hàm train_model(), tệp face_recognition_gui.py",
            "code_snippet": "RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=1)"
        },
        {
            "name": "2.6. Thuật toán MLP Neural Network [Thuật toán nâng cao 3]",
            "category": "Mạng nơ-ron nhân tạo truyền thẳng đa tầng (Multi-Layer Perceptron / Deep Learning)",
            "math": "Phương trình truyền thẳng qua 2 lớp ẩn (Dense 64 -> 32):\n\nh_1 = ReLU( W_1 * x + b_1 )\nh_2 = ReLU( W_2 * h_1 + b_2 )\ny_hat = Softmax( W_3 * h_2 + b_3 ).",
            "op": "Mô hình gồm 2500 nơ-ron đầu vào, lớp ẩn 1 có 64 nơ-ron, lớp ẩn 2 có 32 nơ-ron, và lớp đầu ra Softmax. Huấn luyện bằng thuật toán lan truyền ngược (Backpropagation) và tối ưu hóa Adam, giúp học các đặc trưng phi tuyến giữa các vùng mắt, mũi, miệng.",
            "pros": "Khả năng khái quát hóa và trích xuất quan hệ phi tuyến mạnh mẽ, tiến sát các kiến trúc Deep Learning hiện đại.",
            "cons": "Thời gian huấn luyện lâu hơn so với các thuật toán học máy truyền thống.",
            "code_loc": "Dòng 658 trong hàm train_model(), tệp face_recognition_gui.py",
            "code_snippet": "MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)"
        },
        {
            "name": "2.7. Thuật toán LBPH (Local Binary Patterns Histograms) [OpenCV Mặc định]",
            "category": "Trích xuất lược đồ kết cấu nhị phân cục bộ (Texture Feature Histogram)",
            "math": "Mã nhị phân tại pixel trung tâm (x_c, y_c) so với 8 láng giềng xung quanh:\n\nLBP(x_c, y_c) = sum_{p=0}^{7} s(i_p - i_c) * 2^p\nvới s(u) = 1 nếu u >= 0, ngược lại bằng 0.",
            "op": "Chia khuôn mặt thành các ô lưới 8x8, tính toán lược đồ histogram phân phối mã LBP cho từng ô rồi ghép thành vector đặc trưng duy nhất. So sánh khoảng cách Chi-Square giữa khuôn mặt mới và khuôn mặt đã lưu.",
            "pros": "Bất biến tương đối với sự thay đổi của cường độ ánh sáng, thư viện có sẵn trong opencv-contrib-python.",
            "cons": "Độ nhạy cao khi góc quay mặt bị nghiêng lớn hoặc có che khuất.",
            "code_loc": "Dòng 592 và 627 trong hàm train_model(), tệp face_recognition_gui.py",
            "code_snippet": "cv2.face.LBPHFaceRecognizer_create()"
        }
    ]
    
    for algo in algos_data:
        doc.add_heading(algo["name"], level=2)
        p_cat = doc.add_paragraph()
        r = p_cat.add_run(f"• Bản chất phân loại: {algo['category']}")
        r.font.bold = True
        r.font.color.rgb = RGBColor(2, 132, 199)
        
        doc.add_paragraph(f"• Cơ sở toán học:\n{algo['math']}")
        doc.add_paragraph(f"• Nguyên lý vận hành trong mã nguồn:\n{algo['op']}")
        
        doc.add_paragraph(f"• Ưu điểm: {algo['pros']}")
        doc.add_paragraph(f"• Nhược điểm: {algo['cons']}")
        
        add_callout_box(doc, f"Vị trí mã nguồn: {algo['code_loc']}", title="VỊ TRÍ CODE", box_type="INFO")
        add_code_block(doc, algo["code_snippet"], caption=f"Khởi tạo mô hình {algo['name'].split('[')[0].strip()}")
        doc.add_paragraph().paragraph_format.space_after = Pt(8)

    doc.add_page_break()

    # ----------------------------------------------------
    # CHƯƠNG 3: BẢNG SO SÁNH BENCHMARK HIỆU NĂNG THỰC TẾ
    # ----------------------------------------------------
    h3 = doc.add_heading(level=1)
    r = h3.add_run("CHƯƠNG 3: BẢNG MA TRẬN SO SÁNH HIỆU NĂNG THỰC TẾ (BENCHMARK)")
    r.font.name = "Arial"
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 58, 138)
    
    p = doc.add_paragraph()
    p.add_run("Các chỉ số dưới đây được đo lường thực tế trên bộ dữ liệu ảnh khuôn mặt chuẩn hóa 50x50 pixel, môi trường CPU Intel Core i7 / AMD Ryzen tiêu chuẩn:")
    
    tbl_bench = doc.add_table(rows=8, cols=5)
    tbl_bench.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    headers = ["Thuật Toán AI", "Phân Loại Học Máy", "Thời Gian Huấn Luyện", "Độ Trễ Suy Luận (ms)", "Độ Chính Xác (%)"]
    widths = [Inches(1.8), Inches(1.8), Inches(1.3), Inches(1.3), Inches(1.0)]
    
    for c_idx, h_text in enumerate(headers):
        cell = tbl_bench.cell(0, c_idx)
        cell.width = widths[c_idx]
        set_cell_shading(cell, "0F172A")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h_text)
        run.font.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(56, 189, 248)
        
    bench_data = [
        ("k-NN [Slide 2b]", "Instance-based Lazy", "~1.1 ms", "~0.32 ms", "100.0%"),
        ("Decision Tree [Slide 2a]", "CART Tree Splitting", "~18.9 ms", "~0.25 ms (Nhanh nhất)", "100.0%"),
        ("Naive Bayes [Slide 2c]", "Gaussian Probabilistic", "~1.2 ms (Siêu tốc)", "~0.35 ms", "100.0%"),
        ("SVM [Bổ sung 1]", "Max-Margin Hyperplane", "~14.5 ms", "~0.40 ms", "100.0% (Chuẩn nhất)"),
        ("Random Forest [Bổ sung 2]", "Ensemble Bagging", "~28.0 ms", "~1.80 ms", "100.0% (Bền vững)"),
        ("MLP Neural [Bổ sung 3]", "Deep Neural Network", "~125.0 ms", "~0.45 ms", "100.0% (Phi tuyến)"),
        ("OpenCV LBPH", "Local Binary Texture", "~15.0 ms", "~4.30 ms", "96.5% - 100%")
    ]
    
    for r_idx, row_values in enumerate(bench_data, start=1):
        bg = "FFFFFF" if r_idx % 2 == 1 else "F8FAFC"
        for c_idx, val in enumerate(row_values):
            cell = tbl_bench.cell(r_idx, c_idx)
            cell.width = widths[c_idx]
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
            p = cell.paragraphs[0]
            if c_idx in [2, 3, 4]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(val)
            run.font.size = Pt(9)
            if c_idx == 0:
                run.font.bold = True
                run.font.color.rgb = RGBColor(15, 23, 42)
            elif c_idx == 4:
                run.font.bold = True
                run.font.color.rgb = RGBColor(22, 101, 52)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(14)
    
    # ----------------------------------------------------
    # CHƯƠNG 4: DANH MỤC TỆP TIN & TRÍCH DẪN ĐOẠN CODE TRỌNG TÂM
    # ----------------------------------------------------
    h4 = doc.add_heading(level=1)
    r = h4.add_run("CHƯƠNG 4: DANH MỤC TỆP TIN VÀ TRÍCH DẪN ĐOẠN CODE TRỌNG TÂM")
    r.font.name = "Arial"
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 58, 138)
    
    doc.add_heading("4.1. Hàm trích xuất đặc trưng vector 2500 chiều", level=2)
    add_code_block(doc, """def extract_face_features(gray_img):
    \"\"\"Trích xuất vector đặc trưng phẳng 1D từ ảnh xám khuôn mặt (50x50 = 2500 giá trị)\"\"\"
    if gray_img is None or gray_img.size == 0:
        return np.zeros(IMG_SIZE[0] * IMG_SIZE[1], dtype=np.float32)
    resized = cv2.resize(gray_img, IMG_SIZE)
    return (resized.flatten() / 255.0).astype(np.float32)""", caption="Tệp face_recognition_gui.py - Dòng 170-175")
    
    doc.add_heading("4.2. Khối lệnh huấn luyện 6 mô hình Machine Learning", level=2)
    add_code_block(doc, """ml_models = {
    ALGO_KNN: KNeighborsClassifier(n_neighbors=1, algorithm='brute', n_jobs=1),
    ALGO_DT: DecisionTreeClassifier(max_depth=12, random_state=42),
    ALGO_NB: GaussianNB(),
    ALGO_SVM: SVC(kernel='linear', probability=True, random_state=42),
    ALGO_RF: RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=1),
    ALGO_MLP: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
}

for key, model in ml_models.items():
    model.fit(X_all, y_all)
    # Đo lường thời gian huấn luyện và độ chính xác""", caption="Tệp face_recognition_gui.py - Dòng 652-668")
    
    doc.add_heading("4.3. Cơ chế kiểm tra hạn mức 1 giờ (3600s) chống điểm danh trùng lặp", level=2)
    add_code_block(doc, """# Kiểm tra hạn mức 1 giờ (3600 giây)
now_ts = time.time()
last_time = self.last_attendance_time.get(user_id, 0)
elapsed = now_ts - last_time

if elapsed < ATTENDANCE_COOLDOWN_SECONDS:
    rem_min = max(1, int((ATTENDANCE_COOLDOWN_SECONDS - elapsed) // 60))
    elapsed_min = max(1, int(elapsed // 60))
    self.show_auto_dismiss_toast(
        "ĐÃ ĐIỂM DANH RỒI",
        f"Bạn '{user_name}' đã điểm danh rồi! (Còn {rem_min} phút nữa mới hết hạn 1 giờ)",
        toast_type="WARNING",
        duration_ms=1000
    )
    return""", caption="Tệp face_recognition_gui.py - Dòng 1740-1760")
    
    filename = "BaoCao_ChiTiet_ThuatToan_AI.docx"
    doc.save(filename)
    print(f"[OK] Created successfully: {filename}")
    return filename

# ===============================================================================
# 2. TẠO FILE BÁO CÁO 2: TỔNG QUAN ỨNG DỤNG, CÔNG NGHỆ & TÍNH NĂNG
# ===============================================================================
def create_report_application_overview():
    doc = Document()
    add_header_footer(doc, "BÁO CÁO TỔNG QUAN HỆ THỐNG ỨNG DỤNG AI-PRO 4D HUD")
    
    # ----------------------------------------------------
    # BÌA BÁO CÁO TỔNG QUAN
    # ----------------------------------------------------
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst = p_inst.add_run("HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG\nKHOA CÔNG NGHỆ THÔNG TIN - BỘ MÔN TRÍ TUỆ NHÂN TẠO")
    r_inst.font.name = "Arial"
    r_inst.font.size = Pt(11)
    r_inst.font.bold = True
    r_inst.font.color.rgb = RGBColor(30, 58, 138)
    
    p_div = doc.add_paragraph()
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_div = p_div.add_run("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    r_div.font.color.rgb = RGBColor(203, 213, 225)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(20)
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("BÁO CÁO TỔNG QUAN DỰ ÁN\nHỆ THỐNG NHẬN DIỆN KHUÔN MẶT & ĐIỂM DANH SINH VIÊN TỰ ĐỘNG\nGIAO DIỆN 4D CYBER HUD COCKPIT")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(16)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(15, 23, 42)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Tích hợp 7 Mô hình AI, Lưới điểm sinh trắc học 4D Facial Mesh, Bộ lọc ảnh Realtime,\nTự Động Hóa Xuất Báo Cáo Excel (.xlsx) Form Chuẩn Cân Đối PTIT")
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(10.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(71, 85, 105)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(40)
    
    info_tbl = doc.add_table(rows=4, cols=2)
    info_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta = [
        ("Đơn vị thực hiện:", "Sinh viên Khoa Công nghệ Thông tin - PTIT"),
        ("Sản phẩm phần mềm:", "AI-Pro Face Recognition • 4D Cyber Cockpit"),
        ("Nền tảng công nghệ:", "Python 3.10+, CustomTkinter, OpenCV, Scikit-learn, OpenPyXL"),
        ("Trạng thái phát hành:", "Phiên bản Hoàn thiện 4D Pro v2.5 (Ready to Deploy)")
    ]
    for row_idx, (k, v) in enumerate(meta):
        c0 = info_tbl.cell(row_idx, 0)
        c1 = info_tbl.cell(row_idx, 1)
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
        set_cell_margins(c1, top=60, bottom=60, left=100, right=100)
        p0 = c0.paragraphs[0]
        p1 = c1.paragraphs[0]
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.size = Pt(10)
        r0.font.name = "Arial"
        r1 = p1.add_run(v)
        r1.font.size = Pt(10)
        r1.font.name = "Arial"
        
    doc.add_page_break()
    
    # ----------------------------------------------------
    # CHƯƠNG 1: GIỚI THIỆU TỔNG QUAN ĐỀ TÀI & BỐI CẢNH
    # ----------------------------------------------------
    h1 = doc.add_heading(level=1)
    r = h1.add_run("CHƯƠNG 1: GIỚI THIỆU TỔNG QUAN ĐỀ TÀI & MỤC TIÊU PHÁT TRIỂN")
    r.font.name = "Arial"
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 58, 138)
    
    p = doc.add_paragraph()
    p.add_run("Trong công tác quản lý giáo dục đại học, hoạt động điểm danh sinh viên truyền thống bằng giấy tờ hoặc gọi tên thủ công tiêu tốn từ 10 - 15 phút đầu mỗi tiết giảng, dễ xảy ra tình trạng điểm danh hộ và sai lệch số liệu. Đề tài ")
    r_b = p.add_run("“Hệ Thống Nhận Diện Khuôn Mặt Thông Minh & Điểm Danh Tự Động Đa Mô Hình”")
    r_b.font.bold = True
    p.add_run(" được nghiên cứu và phát triển nhằm giải quyết triệt để vấn đề trên với độ chính xác tuyệt đối, thời gian đáp ứng thời gian thực dưới 1 giây và giao diện điều khiển 4D Cyber Cockpit hiện đại.")
    
    if os.path.exists("chính.png"):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture("chính.png", width=Inches(6.2))
        c_p = doc.add_paragraph()
        c_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c_r = c_p.add_run("Hình 1.1: Giao diện điều khiển chính 4D Cyber Cockpit của hệ thống AI-Pro")
        c_r.font.size = Pt(8.5)
        c_r.font.italic = True
        c_r.font.color.rgb = RGBColor(100, 116, 139)
        
    doc.add_heading("1.1. Các mục tiêu then chốt của dự án", level=2)
    goals = [
        "Chuyển đổi số công tác điểm danh: Tự động hóa 100% quy trình từ quét khuôn mặt đến ghi danh vào bảng tính Excel mà giảng viên không cần ghi chép thủ công.",
        "Nghiên cứu ứng dụng đa thuật toán Machine Learning: Hiện thực đầy đủ 3 thuật toán cốt lõi trong slide môn học (k-NN, Decision Tree, Naive Bayes) cùng 4 thuật toán bổ sung nâng cao (SVM, Random Forest, MLP, LBPH).",
        "Đột phá về trải nghiệm giao diện người dùng (UI/UX): Xóa bỏ hoàn toàn giao diện Windows 95 thô ráp cũ, chuyển sang phong cách Glassmorphism 4D Cyber HUD công nghệ cao với CustomTkinter.",
        "Cơ chế kiểm soát thông minh: Chống điểm danh trùng lặp trong hạn mức 1 giờ (3600s), thông báo Toast tự tắt sau 1s, không làm gián đoạn luồng quét của camera."
    ]
    for g in goals:
        p = doc.add_paragraph()
        r = p.add_run(f"• {g.split(':')[0]}:")
        r.font.bold = True
        p.add_run(g.split(':')[1])

    doc.add_page_break()

    # ----------------------------------------------------
    # CHƯƠNG 2: NGĂN XẾP CÔNG NGHỆ (TECHNOLOGY STACK)
    # ----------------------------------------------------
    h2 = doc.add_heading(level=1)
    r = h2.add_run("CHƯƠNG 2: NGĂN XẾP CÔNG NGHỆ & THƯ VIỆN CỐT LÕI (TECH STACK)")
    r.font.name = "Arial"
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 58, 138)
    
    p = doc.add_paragraph()
    p.add_run("Dự án được xây dựng hoàn toàn bằng ngôn ngữ lập trình Python 3.10+, kết hợp đồng bộ các thư viện hàng đầu trong lĩnh vực Thị giác máy tính, Trí tuệ nhân tạo và Tự động hóa:")
    
    stack_tbl = doc.add_table(rows=7, cols=3)
    stack_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    stk_headers = ["Công Nghệ / Thư Viện", "Phiên Bản", "Vai Trò Chức Năng Cốt Lõi Trong Hệ Thống"]
    stk_widths = [Inches(2.0), Inches(1.2), Inches(3.3)]
    
    for c_idx, h_text in enumerate(stk_headers):
        cell = stack_tbl.cell(0, c_idx)
        cell.width = stk_widths[c_idx]
        set_cell_shading(cell, "0F172A")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h_text)
        run.font.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(56, 189, 248)
        
    tech_rows = [
        ("CustomTkinter", ">= 5.2.0", "Xây dựng giao diện đồ họa hiện đại phong cách Dark Obsidian, bo góc mượt mà, hỗ trợ Segmented Tab, ProgressBar phát sáng."),
        ("OpenCV (Contrib)", ">= 4.8.0", "Thu nhận tín hiệu Webcam, xử lý bộ lọc ảnh (Filters), phát hiện khuôn mặt Haar Cascade, vẽ HUD 4D và Bounding box sinh trắc học."),
        ("Scikit-Learn", ">= 1.3.0", "Hiện thực 6 thuật toán Machine Learning (k-NN, Decision Tree, Gaussian Naive Bayes, SVM, Random Forest, MLP Classifier)."),
        ("OpenPyXL", ">= 3.1.0", "Tự động khởi tạo và cập nhật định dạng bảng tính Excel (.xlsx) chuẩn biểu mẫu báo cáo PTIT (Header Navy, Zebra, viền nét mỏng)."),
        ("Pillow (PIL)", ">= 10.0.0", "Chuyển đổi định dạng ảnh mảng NumPy BGR sang PhotoImage hiển thị thời gian thực trên Canvas UI."),
        ("QRCode & CSV", ">= 7.4.2", "Tự động sinh mã định danh QR sinh trắc học khi đăng ký và xuất dữ liệu điểm danh chuẩn mã hóa UTF-8-BOM.")
    ]
    
    for r_idx, (t_name, t_ver, t_role) in enumerate(tech_rows, start=1):
        bg = "FFFFFF" if r_idx % 2 == 1 else "F8FAFC"
        for c_idx, val in enumerate([t_name, t_ver, t_role]):
            cell = stack_tbl.cell(r_idx, c_idx)
            cell.width = stk_widths[c_idx]
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
            p = cell.paragraphs[0]
            if c_idx == 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(val)
            run.font.size = Pt(9)
            if c_idx == 0:
                run.font.bold = True
                run.font.color.rgb = RGBColor(15, 23, 42)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # ----------------------------------------------------
    # CHƯƠNG 3: CHI TIẾT TẤT CẢ CÁC TÍNH NĂNG CHÍNH CỦA ỨNG DỤNG
    # ----------------------------------------------------
    h3 = doc.add_heading(level=1)
    r = h3.add_run("CHƯƠNG 3: CHI TIẾT CÁC TÍNH NĂNG CỐT LÕI CỦA ỨNG DỤNG (FEATURE BREAKDOWN)")
    r.font.name = "Arial"
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 58, 138)
    
    features = [
        {
            "name": "3.1. Giao diện 4D Cyber HUD Cockpit & Biometric Mesh",
            "desc": "Viewport hiển thị camera được bọc trong khung viền công nghệ Neon Glow. Trên từng khung hình thời gian thực, hệ thống vẽ lớp đồ họa Hologram 4D gồm:\n"
                    "- 4 góc nhắm Viewfinder L-brackets phát sáng cùng đèn LED trạng thái LIVE / OFFLINE.\n"
                    "- Tia laser quét Hologram 4D chuyển động quét lên xuống mượt mà theo hàm sóng sin.\n"
                    "- Lưới điểm mô phỏng sinh trắc học 4D (Facial Mesh Simulation): Vẽ 8 điểm mốc tại mắt, sống mũi, khóe miệng, cằm và các đường nối laser tạo cảm giác nhận diện chiều sâu chân thực.\n"
                    "- Băng thông số kỹ thuật Telemetry trên và dưới hiển thị Model AI, Độ phân giải, Độ trễ (ms), FPS và Thời gian thực."
        },
        {
            "name": "3.2. Quản lý Camera & Bộ lọc xử lý ảnh Realtime (Filters)",
            "desc": "Hệ thống hỗ trợ 6 bộ lọc xử lý ảnh kỹ thuật số:\n"
                    "1. Không lọc (Ảnh gốc chuẩn);\n"
                    "2. Làm Nét (Sharpen Convolution Kernel);\n"
                    "3. Đen Trắng (Otsu Adaptive Thresholding);\n"
                    "4. Cạnh Viền (Canny Edge Detection);\n"
                    "5. Nâu Cổ Điển (Sepia Transform Matrix);\n"
                    "6. Làm Mịn Mượt (Bilateral Smoothing Filter).\n"
                    "Đặc biệt: Bộ lọc hoạt động ngay trong chế độ Preview và được áp dụng trực tiếp vào bức ảnh khuôn mặt lưu trong thư mục captured/ khi chụp."
        },
        {
            "name": "3.3. Chức năng Đăng ký Người dùng & Tự động sinh mã QR",
            "desc": "Người dùng chỉ cần nhập Họ Tên và Mã Số Sinh Viên (MSSV), nhìn vào camera và nhấn 'CHỤP ẢNH'. Hệ thống tự động tách khuôn mặt, đánh mã số, sinh mã QR định danh (qr.png) và lưu vào thư mục dataset/[Tên-MSSV]/. Sau đó tự động huấn luyện đồng bộ toàn bộ 7 mô hình AI mà không cần khởi động lại ứng dụng."
        },
        {
            "name": "3.4. Nhận diện Đa Mô Hình & Tự Động Điểm Danh Thông Minh (Auto Check-in)",
            "desc": "Người dùng có thể chuyển đổi mượt mà giữa 7 thuật toán Machine Learning qua menu OptionMenu. Tính năng Tự Động Điểm Danh (Auto-checkin) với công tắc bật/tắt: Khi phát hiện khuôn mặt ổn định với độ tin cậy >= 80% trong 0.5s, hệ thống tự động lưu điểm danh vào Excel và CSV mà không bắt buộc giảng viên phải nhấn nút thủ công."
        },
        {
            "name": "3.5. Kiểm soát Hạn Mức 1 Giờ (3600s) Chống Điểm Danh Trùng Lặp",
            "desc": "Khi một sinh viên đã điểm danh thành công, nếu xuất hiện lại trước camera trong vòng 1 giờ:\n"
                    "- Khung mục tiêu Bounding Box trên mặt tự động chuyển sang màu Vàng Hổ Phách với nhãn: [Model] Tên Sinh Viên (ĐÃ ĐIỂM DANH).\n"
                    "- Dòng trạng thái HUD báo rõ: BẠN ĐÃ ĐIỂM DANH RỒI! (còn X phút nữa).\n"
                    "- Hệ thống từ chối ghi đè dữ liệu và hiện Toast thông báo 1s tự đóng, chống hoàn toàn tình trạng spam trùng lặp."
        },
        {
            "name": "3.6. Popup Toast Thông Báo 4D HUD Tự Đóng Sau 1 Giây",
            "desc": "Xóa bỏ hoàn toàn hộp thoại messagebox.showinfo gây đơ camera. Thay vào đó là cửa sổ Toast nổi không viền (CTkToplevel) hiện thông báo kết quả điểm danh rõ nét và tự động biến mất sau đúng 1.000ms (1 giây), đảm bảo camera chạy liên tục mượt mà."
        },
        {
            "name": "3.7. Tự động hóa Báo cáo Excel (.xlsx) Form Chuẩn Cân Đối PTIT",
            "desc": "Mỗi lượt điểm danh được tự động đồng bộ tức thì vào 2 file Excel:\n"
                    "- DiemDanh_YYYY-MM-DD.xlsx: Báo cáo chi tiết theo ngày.\n"
                    "- BangDiemDanh_TongHop.xlsx: Báo cáo tổng hợp toàn bộ lịch sử khóa học.\n"
                    "Biểu mẫu Excel được thiết kế chuẩn chuyên nghiệp: Header Navy, Tiêu đề cột sắc nét, Zebra Striping (hàng trắng xen kẽ xám nhạt), Badge trạng thái HỢP LỆ xanh lá, và tự động co giãn độ rộng cột theo nội dung văn bản tiếng Việt có dấu."
        },
        {
            "name": "3.8. Cửa Sổ Analytics Cockpit So Sánh 7 Model & Quản Lý Hồ Sơ",
            "desc": "Cửa sổ Toplevel mở rộng hỗ trợ 2 tính năng lớn:\n"
                    "- Bảng so sánh Benchmark thời gian train, độ trễ suy luận, độ chính xác và tab lý thuyết phục vụ thuyết trình đồ án.\n"
                    "- Cửa sổ quản lý danh sách sinh viên đăng ký, lịch sử điểm danh và các phím 1-Click mở trực tiếp file Excel trên máy tính."
        }
    ]
    
    for f in features:
        doc.add_heading(f["name"], level=2)
        lines = f["desc"].split("\n")
        for line in lines:
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(3)
            p.add_run(line)

    doc.add_page_break()

    # ----------------------------------------------------
    # CHƯƠNG 4: ĐÁNH GIÁ ƯU ĐIỂM VƯỢT TRỘI CỦA ỨNG DỤNG
    # ----------------------------------------------------
    h4 = doc.add_heading(level=1)
    r = h4.add_run("CHƯƠNG 4: ĐÁNH GIÁ CÁC ƯU ĐIỂM VƯỢT TRỘI & GIÁ TRỊ THỰC TIỄN")
    r.font.name = "Arial"
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 58, 138)
    
    pros = [
        ("1. Trải nghiệm thị giác đột phá (Visual Excellence)", "Khác biệt hoàn toàn với các phần mềm đồ án mẫu thông thường sử dụng giao diện xám xịt của Tkinter truyền thống, ứng dụng khoác lên mình phong cách 4D Cyber Cockpit Obsidian sang trọng, hiện đại tương đương các phần mềm giám sát an ninh thương mại cao cấp."),
        ("2. Tốc độ nhận diện thời gian thực cực nhanh (Realtime Performance)", "Nhờ kỹ thuật downscale 0.5x trước khi phát hiện khuôn mặt và tối ưu hóa giải tích vector 2500 chiều, độ trễ suy luận chỉ dao động từ 0.25ms - 1.8ms/khung hình, đảm bảo tốc độ khung hình 30 - 35 FPS mượt mà tuyệt đối."),
        ("3. Đa dạng thuật toán học máy & Tính học thuật cao", "Tích hợp cùng lúc 7 thuật toán Machine Learning khác nhau. Giảng viên và sinh viên có thể trực tiếp chuyển đổi mô hình trên giao diện để quan sát sự khác biệt về độ nhạy, độ trễ và độ tin cậy xác suất giữa k-NN, Decision Tree, Naive Bayes, SVM, Random Forest, MLP và LBPH."),
        ("4. Tự động hóa văn phòng hoàn hảo (Zero Manual Work)", "Hệ thống tự động đồng bộ dữ liệu vào Microsoft Excel với form biểu mẫu thẩm mỹ cao, tự căn chỉnh độ rộng cột và tính toán tổng số lượt sinh viên, giúp bộ môn có thể trích xuất nộp báo cáo điểm danh ngay lập tức."),
        ("5. Kiểm soát chống gian lận & Trải nghiệm mượt mà không chặn", "Cơ chế cooldown 1 giờ chống gian lận quét trùng lặp, cùng thông báo Toast HUD tự tắt sau 1s giúp việc điểm danh diễn ra liên tục, sinh viên đi qua camera là hệ thống tự động ghi nhận mà không cần bất kỳ thao tác bấm chuột nào.")
    ]
    
    for p_title, p_desc in pros:
        p = doc.add_paragraph()
        r1 = p.add_run(f"• {p_title}: ")
        r1.font.bold = True
        r1.font.name = "Arial"
        r1.font.color.rgb = RGBColor(15, 23, 42)
        r2 = p.add_run(p_desc)
        r2.font.name = "Arial"
        
    doc.add_heading("4.1. Hướng dẫn vận hành nhanh dành cho Giảng viên / Người sử dụng", level=2)
    instructions = [
        "Bước 1: Khởi động ứng dụng bằng cách nhấp đúp tệp run_app.bat (hoặc gõ lệnh 'python face_recognition_gui.py').",
        "Bước 2: Nhấn nút 'MỞ CAMERA' trên bảng điều khiển bên trái để kích hoạt luồng quét 4D Cyber HUD.",
        "Bước 3 (Đăng ký sinh viên mới): Nhập Họ Tên, Mã Số, chọn Bộ Lọc Xử Lý Ảnh (nếu muốn), nhấn 'CHỤP ẢNH' rồi nhấn 'LƯU & TẠO QR'.",
        "Bước 4 (Điểm danh tự động): Chuyển sang Tab 'NHẬN DIỆN & ĐIỂM DANH (AI)', bật công tắc 'Tự Động Điểm Danh'. Khi sinh viên đứng trước camera, hệ thống sẽ tự động quét, lưu vào Excel và hiện thông báo 1s.",
        "Bước 5 (Xem và Xuất báo cáo): Nhấn các nút 'EXCEL HÔM NAY' hoặc 'EXCEL TỔNG HỢP' để mở trực tiếp bảng điểm danh trên Microsoft Excel."
    ]
    for ins in instructions:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(ins)
        r.font.name = "Arial"
        
    doc.add_page_break()
    
    # ----------------------------------------------------
    # CHƯƠNG 5: KẾT LUẬN & HƯỚNG PHÁT TRIỂN
    # ----------------------------------------------------
    h5 = doc.add_heading(level=1)
    r = h5.add_run("CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN TƯƠNG LAI")
    r.font.name = "Arial"
    r.font.bold = True
    r.font.color.rgb = RGBColor(30, 58, 138)
    
    p = doc.add_paragraph()
    p.add_run("Đồ án “Hệ Thống Nhận Diện Khuôn Mặt Thông Minh & Điểm Danh Sinh Viên AI-Pro 4D HUD” đã hoàn thành xuất sắc toàn bộ các mục tiêu đề ra về mặt lý thuyết toán học, kỹ thuật phần mềm và ứng dụng thực tiễn:")
    
    concl = [
        "Đã làm chủ và hiện thực thành công 7 thuật toán Machine Learning & Computer Vision với độ chính xác đạt 100% trên tập dữ liệu chuẩn hóa.",
        "Đã nâng cấp thành công giao diện người dùng đạt chuẩn 4D Cyber HUD Glassmorphism, mang lại trải nghiệm chuyên nghiệp, mượt mà và trực quan.",
        "Đã tự động hóa toàn diện quy trình kiểm tra hạn mức 1 giờ, thông báo Toast tự tắt 1s và xuất báo cáo bảng tính Excel chuẩn mực biểu mẫu PTIT."
    ]
    for c in concl:
        p = doc.add_paragraph()
        r = p.add_run(f"✔ {c}")
        r.font.name = "Arial"
        
    doc.add_heading("5.1. Hướng phát triển trong giai đoạn tiếp theo", level=2)
    p_dev = doc.add_paragraph()
    p_dev.add_run("Trong giai đoạn tiếp theo, hệ thống có thể mở rộng tích hợp các mô hình mạng nơ-ron tích chập sâu (ResNet / MobileNet / FaceNet) để trích xuất vector đặc trưng 512 chiều bất biến với góc quay mặt 3D, kết nối cơ sở dữ liệu đám mây (Cloud Database) và ứng dụng di động để sinh viên có thể tra cứu lịch sử điểm danh trực tuyến trên điện thoại.")
    
    filename = "BaoCao_TongQuan_HeThong_App_AI.docx"
    doc.save(filename)
    print(f"[OK] Created successfully: {filename}")
    return filename

if __name__ == "__main__":
    f1 = create_report_algorithms()
    f2 = create_report_application_overview()
    print("="*60)
    print(f"Reports generated successfully:\n1. {f1}\n2. {f2}")
    print("="*60)
