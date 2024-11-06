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
        if not os.path.exists(model_path):
            st.error(f"ไม่พบไฟล์โมเดลที่: {model_path}")
            return None
            
        model = load_model(model_path)
        st.success("โหลดโมเดลสำเร็จ!")
        return model
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
        return None

# Sidebar for model configuration
with st.sidebar:
    st.header("การตั้งค่า")
    
    # ให้ผู้ใช้เลือกวิธีการโหลดโมเดล
    load_method = st.radio(
        "เลือกวิธีการโหลดโมเดล",
        ["อัพโหลดไฟล์โมเดล", "ระบุเส้นทางไฟล์โมเดล"]
    )
    
    if load_method == "อัพโหลดไฟล์โมเดล":
        uploaded_model = st.file_uploader("อัพโหลดไฟล์โมเดล", type=['h5'])
        if uploaded_model:
            # บันทึกไฟล์ที่อัพโหลดไว้ชั่วคราว
            temp_model_path = "temp_model.h5"
            with open(temp_model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.session_state.model_path = temp_model_path
    else:
        default_path = os.path.join(os.getcwd(), 'modellittle.h5')
        model_path = st.text_input("ระบุเส้นทางไฟล์โมเดล", value=default_path)
        if model_path:
            st.session_state.model_path = model_path

    # ปุ่มโหลดโมเดล
    if st.button("โหลดโมเดล"):
        if 'model_path' in st.session_state:
            model = load_keras_model(st.session_state.model_path)
            if model is not None:
                st.session_state.model = model

# Class labels
class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TB']

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Main content
st.markdown('<p class="big-font">Medical Image Classification</p>', unsafe_allow_html=True)

# Page navigation
page = st.selectbox("Select Page", ["Home", "Upload", "Feed"])

if page == "Home":
    st.write("ยินดีต้อนรับสู่แอพพลิเคชันวิเคราะห์ภาพทางการแพทย์!")
    st.markdown("""
    ### เกี่ยวกับแอพพลิเคชัน
    แอพพลิเคชันนี้ใช้ Deep Learning ในการจำแนกภาพเอกซเรย์ทรวงอกเป็น 4 ประเภท:
    - COVID-19
    - ปกติ (Normal)
    - ปอดอักเสบ (Pneumonia)
    - วัณโรค (TB)
    
    ### วิธีใช้งาน
    1. เลือกวิธีการโหลดโมเดลในแถบด้านซ้าย (ไฟล์ .h5)
    2. โหลดโมเดลให้เรียบร้อย
    3. ไปที่หน้า "Upload" เพื่ออัพโหลดภาพที่ต้องการวิเคราะห์
    4. รอผลการวิเคราะห์
    
    ### หมายเหตุ
    กรุณาตรวจสอบว่าภาพที่อัพโหลดเป็นภาพเอกซเรย์ทรวงอกที่ชัดเจน
    """)

elif page == "Upload":
    if 'model' not in st.session_state:
        st.warning("กรุณาโหลดโมเดลก่อนใช้งาน โดยเลือกวิธีการโหลดโมเดลในแถบด้านซ้าย")
    else:
        st.header("อัพโหลดภาพ")
        
        uploaded_file = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Load and preprocess the image
                img = load_img(BytesIO(uploaded_file.read()), target_size=(400, 400))
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = x.astype('float32') / 255.0
                
                # แสดงภาพและผลการวิเคราะห์
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="ภาพที่อัพโหลด", use_column_width=True)
                
                # วิเคราะห์ภาพ
                predictions = st.session_state.model.predict(x)
                
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class]) * 100
                result_label = class_labels[predicted_class]
                
                # แสดงผลการวิเคราะห์
                with col2:
                    st.markdown("### ผลการวิเคราะห์")
                    st.markdown(f"**การวินิจฉัย:** {result_label}")
                    st.markdown(f"**ความมั่นใจ:** {confidence:.2f}%")
                    
                    st.markdown("### ระดับความมั่นใจสำหรับทุกประเภท")
                    for i, (label, prob) in enumerate(zip(class_labels, predictions[0])):
                        st.progress(float(prob))
                        st.write(f"{label}: {float(prob)*100:.2f}%")
                    
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")
                st.write("กรุณาตรวจสอบว่าไฟล์ภาพถูกต้อง")

elif page == "Feed":
    st.header("ข้อมูลและสถิติ")
    st.write("""
    ### ข้อมูลโมเดล
    - ขนาดภาพอินพุต: 400x400 pixels
    - ประเภทโมเดล: Keras Deep Learning Model
    - จำนวนคลาส: 4
    
    ### ข้อมูลโรค
    1. **COVID-19**
       - การติดเชื้อไวรัสในระบบทางเดินหายใจ
       - มีลักษณะ ground-glass opacity ในภาพเอกซเรย์ทรวงอก
    
    2. **ปอดอักเสบ (Pneumonia)**
       - การติดเชื้อที่ทำให้เกิดการอักเสบในถุงลม
       - อาจเกิดจากไวรัสหรือแบคทีเรีย
    
    3. **วัณโรค (TB)**
       - การติดเชื้อแบคทีเรียที่ส่งผลต่อปอดเป็นหลัก
       - แสดงรูปแบบเฉพาะในภาพเอกซเรย์ทรวงอก
    
    ### คำเตือน
    เครื่องมือนี้ใช้เพื่อการศึกษาเท่านั้น กรุณาปรึกษาบุคลากรทางการแพทย์สำหรับการวินิจฉัยทางการแพทย์เสมอ
    """)