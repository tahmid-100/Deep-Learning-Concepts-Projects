import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("cat_dog_model.h5")

# Title
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image, and the model will predict if it's a cat or a dog.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction function
def predict(image):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    image = cv2.resize(image, (100, 100))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# Show image and prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict(image)

    st.markdown(f"### Prediction: {label}")
    st.markdown(f"**Confidence:** {confidence:.2%}")
