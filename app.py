import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Set page configuration
st.set_page_config(page_title="🧠 Handwritten Digit Recognizer", layout="centered")

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

# Title
st.title("🧠 Handwritten Digit Recognizer")
st.markdown("🎯 **Recognize digits (0–9) drawn or uploaded by you!**")

# Option to draw or upload
option = st.radio("✍️ Choose input method:", ["🖌️ Draw Digit", "📁 Upload Image"])

if option == "🖌️ Draw Digit":
    st.markdown("🎨 **Draw a digit (0-9) below:**")

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

    if st.button("🔍 Predict from Drawing"):
        if canvas_result.image_data is not None:
            img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
            processed = preprocess(img)
            prediction = model.predict(processed)
            st.success(f"✅ **Predicted Digit:** `{np.argmax(prediction)}` 🔢")

elif option == "📁 Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image of a digit (ideally 28x28 or larger)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", width=150)

        if st.button("🔍 Predict from Upload"):
            processed = preprocess(image)
            prediction = model.predict(processed)
            st.success(f"✅ **Predicted Digit:** `{np.argmax(prediction)}` 🔢")
