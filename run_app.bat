@echo off
chcp 65001 >nul
title HE THONG NHAN DIEN KHUON MAT AI-PRO (PTIT)
color 0A
cls

echo ===============================================================================
echo     HOC VIEN CONG NGHE BUU CHINH VIEN THONG (PTIT)
echo     DO AN MON HOC: TRI TUE NHAN TAO - AI NHAN DIEN KHUON MAT
echo ===============================================================================
echo     Tich hop 7 thuat toan Machine Learning va Computer Vision:
echo       1. [Slide 2a] Cay quyet dinh (Decision Tree)
echo       2. [Slide 2b] k-Lang gieng gan nhat (k-NN)
echo       3. [Slide 2c] Phan lop Bayes tho ngay (Gaussian Naive Bayes)
echo       4. [Bo sung 1] May hoc vector ho tro (Support Vector Machine - SVM)
echo       5. [Bo sung 2] Rung ngau nhien (Random Forest Ensemble)
echo       6. [Bo sung 3] Mang no-ron nhan tao da tang (MLP Neural Network)
echo       7. [OpenCV] Mau nhi phan cuc bo (LBPH Face Recognizer)
echo ===============================================================================
echo.

echo [1/3] Kiem tra moi truong Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [LOI] Khong tim thay Python trong he thong!
    echo Vui long cai dat Python 3.9+ va tick chon "Add Python to PATH".
    echo Trang chu: https://www.python.org/
    echo.
    pause
    exit /b 1
)
python --version

echo.
echo [2/3] Kiem tra thu vien phan mem (requirements.txt)...
pip install -r requirements.txt --quiet --disable-pip-version-check
echo   OK: Thu vien da san sang!

echo.
echo [3/3] Dang khoi dong Giao Dien AI-PRO (face_recognition_gui.py)...
echo ===============================================================================
echo.

python face_recognition_gui.py

if %errorlevel% neq 0 (
    echo.
    color 0C
    echo [THONG BAO] Ung dung gap loi voi ma thoat: %errorlevel%
    pause
)
