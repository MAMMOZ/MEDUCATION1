import streamlit as st
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from io import BytesIO
import os

import requests
import urllib

data = "15dKsgaGl4Ywq0YwMJsV0QmjAEenwg6Jk"

uuid=requests.get(f"https://drive.usercontent.google.com/download?id={data}&export=download&authuser=0",headers={
                "Cache-Control":"max-age=0",
                "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}).text.split('uuid" value="')[1].split('">')[0]
def a():
    urllib.request.urlretrieve(f"https://drive.usercontent.google.com/download?id={data}&export=download&authuser=0&confirm=t&uuid={uuid}", f"final_model.h5")
a()

app = Flask(__name__)

current_directory = os.getcwd()

# ใช้ os.path.join เพื่อสร้างเส้นทางที่เป็นกลางกับระบบปฏิบัติการ
# โหลดโมเดลของคุณ
model_path = os.path.join(current_directory, 'final_model.h5')

model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TB']

# Streamlit app title
st.title("Medical Image Classification")

# Page navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Upload", "Feed"])

if page == "Home":
    st.write("Welcome to the Medical Image Classification app!")

elif page == "Upload":
    st.header("Upload an Image")
    
    # Upload file
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and preprocess the image
        img = load_img(BytesIO(uploaded_file.read()), target_size=(400, 400))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalization
        
        # Display uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Predict the result
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        result_label = class_labels[predicted_class]
        
        # Show the result
        st.write(f"Prediction: {result_label}")
        st.write(f"Confidence: {confidence:.2f}%")

elif page == "Feed":
    st.header("Feed")
    st.write("This page could be used to show feed data, statistics, or other relevant information.")

