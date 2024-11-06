import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from io import BytesIO
import os

# Set page configuration
st.set_page_config(
    page_title="Medical Image Classification",
    page_icon="🏥",
    layout="wide"
)

# ฟังก์ชันสำหรับโหลดโมเดล
def load_keras_model(model_path):
    try:
        model = load_model(model_path)
        st.success("โหลดโมเดลสำเร็จ!")
        return model
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
        return None

# Sidebar for model configuration
with st.sidebar:
    st.header("การตั้งค่า")
    
    load_method = st.radio(
        "เลือกวิธีการโหลดโมเดล",
        ["อัพโหลดไฟล์โมเดล", "ระบุเส้นทางไฟล์โมเดล"]
    )
    
    if load_method == "อัพโหลดไฟล์โมเดล":
        uploaded_model = st.file_uploader("อัพโหลดไฟล์โมเดล", type=['h5'])
        if uploaded_model:
            with open("temp_model.h5", "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.session_state.model_path = "temp_model.h5"
            st.session_state.model = load_keras_model(st.session_state.model_path)
    else:
        default_path = os.path.join(os.getcwd(), 'modellittle.h5')
        model_path = st.text_input("ระบุเส้นทางไฟล์โมเดล", value=default_path)
        if model_path:
            st.session_state.model_path = model_path
            st.session_state.model = load_keras_model(st.session_state.model_path)

    # ปุ่มโหลดโมเดล
    if st.button("โหลดโมเดล"):
        if 'model_path' in st.session_state:
            st.session_state.model = load_keras_model(st.session_state.model_path)

# ตรวจสอบว่ามีโมเดลอยู่ใน session_state แล้วหรือไม่
if 'model' not in st.session_state:
    st.warning("กรุณาโหลดโมเดลก่อนใช้งาน โดยเลือกวิธีการโหลดโมเดลในแถบด้านซ้าย")

# Class labels
class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TB']

# Main content
page = st.selectbox("Select Page", ["Home", "Upload", "Feed"])

if page == "Upload":
    if 'model' not in st.session_state:
        st.warning("กรุณาโหลดโมเดลก่อนใช้งาน โดยเลือกวิธีการโหลดโมเดลในแถบด้านซ้าย")
    else:
        st.header("อัพโหลดภาพ")
        uploaded_file = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                img = load_img(BytesIO(uploaded_file.read()), target_size=(400, 400))
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = x.astype('float32') / 255.0
                
                # วิเคราะห์ภาพ
                predictions = st.session_state.model.predict(x)
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class]) * 100
                result_label = class_labels[predicted_class]
                
                st.image(img, caption="ภาพที่อัพโหลด", use_column_width=True)
                st.markdown(f"**การวินิจฉัย:** {result_label}")
                st.markdown(f"**ความมั่นใจ:** {confidence:.2f}%")
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")