FROM python:3.10-slim

# ติดตั้ง dependency ที่จำเป็น
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดไปยัง container
COPY . .

# ตั้งค่าเริ่มต้น
CMD ["gunicorn", "app:app"]
