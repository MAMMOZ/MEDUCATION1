import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
import streamlit as st
import os

# ตั้งค่าโมเดล
current_directory = os.getcwd()
model_path = os.path.join(current_directory, 'final_model.h5')
model = tf.keras.models.load_model(model_path)

# คำอธิบายของคลาสที่โมเดลทำนายได้
class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TB']

# ฟังก์ชันสำหรับการพยากรณ์
def predict_image(uploaded_file):
    img = load_img(uploaded_file, target_size=(400, 400))  # ปรับขนาดภาพ
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)  # เพิ่ม dimension
    x = x / 255.0  # Normalization

    # ทำนายผลลัพธ์
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # ค่าความมั่นใจ
    result_label = class_labels[predicted_class]
    
    return result_label, confidence

# สร้างอินเทอร์เฟซใน Streamlit
st.title("COVID-19 Image Classification")
st.write("Upload an X-ray image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงภาพที่อัปโหลด
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # ทำนายผลลัพธ์
    label, confidence = predict_image(uploaded_file)
    
    # แสดงผลลัพธ์
    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")
