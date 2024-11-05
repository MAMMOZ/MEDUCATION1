import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
import os
import streamlit as st
from tensorflow.keras.models import load_model

# Load the model
current_directory = os.getcwd()
model_path = os.path.join(current_directory, 'final_model.h5')
if os.path.exists(model_path):
    model = load_model(model_path)
    print("ammoz")
else:
    print(f"Error: Model file '{model_path}' not found!")

# Define the class labels
class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TB']

def predict_image(image):
    # Preprocess the image
    img = load_img(image, target_size=(400, 400))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Make the prediction
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return class_labels[predicted_class], confidence

# Streamlit app
st.title("Medical Image Classification")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Classify the image
    result, confidence = predict_image(uploaded_file)
    st.write(f"Predicted class: {result}")
    st.write(f"Confidence: {confidence:.2f}%")