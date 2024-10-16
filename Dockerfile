# ใช้ภาพฐานจาก Python 3.9
FROM python:3.9-slim

# ตั้งค่าโฟลเดอร์ทำงาน
WORKDIR /app

# คัดลอกไฟล์ requirements.txt และไฟล์แอปไปยัง container
COPY requirements.txt .

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดแอปพลิเคชัน
COPY . .

# ตั้งค่า PORT ที่แอปจะใช้
EXPOSE 5000

# คำสั่งในการรันแอปพลิเคชัน
CMD ["python", "app.py"]
