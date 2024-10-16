import streamlit as st
import requests
import subprocess
import os
import time

# ฟังก์ชันในการรัน Flask app
def run_flask():
    # รัน app.py ใน subprocess
    subprocess.Popen(['python', 'app.py'])

# ตรวจสอบว่ามี Flask รันอยู่หรือไม่
def is_flask_running():
    try:
        response = requests.get("http://127.0.0.1:5000")
        return response.status_code == 200
    except requests.ConnectionError:
        return False

# เริ่ม Flask app ถ้ายังไม่ได้รัน
if not is_flask_running():
    run_flask()
    # รอให้ Flask มีเวลาในการเริ่มต้น
    time.sleep(3)  # สามารถปรับเวลาได้ตามต้องการ

# ตั้งชื่อหัวข้อใน Streamlit
st.title("Image Classification with Flask")

# สร้างตัวอัปโหลดไฟล์
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# เมื่อกดปุ่ม "Predict"
if st.button("Predict") and uploaded_file:
    # สร้าง payload สำหรับการส่งภาพไปยัง Flask
    files = {'image': uploaded_file}

    # ส่งคำขอไปยัง Flask API
    response = requests.post("http://127.0.0.1:5000/predict", files=files)

    # ตรวจสอบผลลัพธ์
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}%")
    else:
        st.error(response.json().get("error", "Error during prediction"))
