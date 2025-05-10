import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Set page configuration
st.set_page_config(page_title="🧠 Handwritten Digit Recognizer", layout="centered")

# Custom CSS for animations and background
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #74ebd5, #ACB6E5);
    }
    .stApp {
        background-image: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
        background-attachment: fixed;
        background-size: cover;
    }
    h1 {
        color: transparent;
        background: linear-gradient(to right, #f12711, #f5af19);
        -webkit-background-clip: text;
        font-size: 3em;
        animation: pulse 2s infinite;
        text-align: center;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .celebrate {
        text-align: center;
        font-size: 2em;
        color: #ff4081;
        animation: pop 0.6s ease-in-out;
    }
    @keyframes pop {
        0% { transform: scale(0.5); opacity: 0; }
        100% { transform: scale(1.2); opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

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

# Title with animated gradient
st.title("🧠 Handwritten Digit Recognizer")
st.markdown("🎯 **Recognize digits (0–9) drawn or uploaded by you!**")

# Input method selector
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
            predicted_digit = np.argmax(prediction)

            st.success(f"✅ **Predicted Digit:** `{predicted_digit}` 🔢")
            st.balloons()

            # Celebratory text
            st.markdown('<div class="celebrate">🎉 Woohoo! Great job! 🎉</div>', unsafe_allow_html=True)

            # Confetti GIF
            st.image("https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif", width=250)

elif option == "📁 Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image of a digit (ideally 28x28 or larger)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", width=150)

        if st.button("🔍 Predict from Upload"):
            processed = preprocess(image)
            prediction = model.predict(processed)
            predicted_digit = np.argmax(prediction)

            st.success(f"✅ **Predicted Digit:** `{predicted_digit}` 🔢")
            st.balloons()

            st.markdown('<div class="celebrate">🎊 You nailed it! 🎊</div>', unsafe_allow_html=True)
            st.image("https://media.giphy.com/media/xT0Gqz3vGq7LiknyGg/giphy.gif", width=250)
