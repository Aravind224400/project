import streamlit as st
st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

# Other imports
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os
from streamlit_drawable_canvas import st_canvas

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model_v2.h5")

model = load_model()

# (Then your preprocess, predict_digit, UI code...)

def preprocess_image(image):
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 28, 28, 1)

def predict_digit(image):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    return digit, confidence, prediction[0]

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🧠 MNIST Handwritten Digit Recognizer")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        margin-right: 5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("ℹ️ Instructions")
st.sidebar.markdown("""
- Draw or upload a digit (0–9)
- Use **white background** and **black digit**
- Larger, centered digits work best
""")

tab1, tab2 = st.tabs(["📤 Upload Image", "✏️ Draw Digit"])

with tab1:
    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file and model:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)

        digit, confidence, probs = predict_digit(image)

        st.markdown(f"### ✅ Predicted: `{digit}`")
        st.progress(confidence / 100)
        st.caption(f"Confidence: {confidence:.2f}%")
        st.bar_chart(probs)

with tab2:
    st.write("Draw a digit below (white canvas, black digit):")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        drawn_image = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
        st.image(drawn_image.resize((100, 100)), caption="Your Drawing", width=100)

        if st.button("🔍 Predict") and model:
            digit, confidence, probs = predict_digit(drawn_image)

            st.markdown(f"### ✅ Predicted: `{digit}`")
            st.progress(confidence / 100)
            st.caption(f"Confidence: {confidence:.2f}%")
            st.bar_chart(probs)

st.markdown("---")
st.info("Tip: Draw clearly and center your digit. The model mimics the MNIST dataset.")
