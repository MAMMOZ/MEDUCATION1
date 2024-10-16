import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
import streamlit as st
import os
from io import BytesIO

# Set the title of the app
st.title("Disease Prediction from X-ray Images")

current_directory = os.getcwd()

# Load your model
model_path = os.path.join(current_directory, 'final_model.h5')
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TB']

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the image using streamlit
    img = load_img(BytesIO(uploaded_file.read()), target_size=(400, 400))
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Prepare the image for prediction
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = x / 255.0  # Normalize

    # Predict the result
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # Confidence score

    result_label = class_labels[predicted_class]

    # Display the prediction results
    st.write(f"Prediction: {result_label}")
    st.write(f"Confidence: {confidence:.2f}%")
