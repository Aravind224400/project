import streamlit as st
# This must be the FIRST Streamlit command
st.set_page_config(page_title="Hand Digit Recognizer", layout="centered")

from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model_v2.h5")

model = load_model()

# Preprocessing function
def preprocess(image):
    image = image.resize((28, 28)).convert('L')  # Grayscale and resize
    image = ImageOps.invert(image)
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

# Streamlit UI
st.title("Hand Digit Recognizer")
st.markdown("Draw a digit (0-9) below:")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
        processed = preprocess(img)
        prediction = model.predict(processed)
        st.success(f"Predicted Digit: {np.argmax(prediction)}")
