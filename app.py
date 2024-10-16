import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from io import BytesIO
import os

app = FastAPI()

# กำหนดเส้นทางในการใช้ไฟล์ static
app.mount("/static", StaticFiles(directory="static"), name="static")

# กำหนด templates
templates = Jinja2Templates(directory="templates")

current_directory = os.getcwd()

# โหลดโมเดลของคุณ
model_path = os.path.join(current_directory, 'final_model.h5')
model = tf.keras.models.load_model(model_path)

# คำอธิบายของคลาสที่โมเดลทำนายได้
class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TB']

@app.get("/", response_class=HTMLResponse)
async def web(request: Request):
    return templates.TemplateResponse("web.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload(request: Request):
    return templates.TemplateResponse("Upload.html", {"request": request})

@app.get("/feed", response_class=HTMLResponse)
async def feed(request: Request):
    return templates.TemplateResponse("feed.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.filename == '':
        return {"error": "Error: No selected file"}

    # โหลดภาพจากไฟล์อัปโหลด โดยใช้ stream เพื่อแปลงเป็น BytesIO
    img = load_img(BytesIO(await file.read()), target_size=(400, 400))  # ปรับขนาดภาพให้ตรงกับขนาดที่โมเดลต้องการ
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)  # เพิ่ม dimension ให้ตรงกับ input ของโมเดล
    x = x / 255.0  # Normalization

    # ทำนายผลลัพธ์
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # ค่าความมั่นใจ

    result_label = class_labels[predicted_class]

    return templates.TemplateResponse("result.html", {"request": {}, "prediction": result_label, "confidence": confidence})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
