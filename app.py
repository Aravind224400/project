import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Set page configuration
st.set_page_config(page_title="🧠 Handwritten Digit Recognizer", layout="centered")

# Initialize score in session
if 'score' not in st.session_state:
    st.session_state.score = 0

# Custom CSS to set the provided aesthetic background image with pure white text and black info text
st.markdown("""
    <style>
    body {
        margin: 0;
        padding: 0;
        background-image: url('https://png.pngtree.com/thumb_back/fw800/background/20190223/ourmid/pngtree-simple-and-intelligent-facial-recognition-advertising-background-backgroundintelligentadvancedlight-spottechnological-senseface-image_82888.jpg');
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
    }

    .stApp {
        background: transparent; /* No overlay, background fully visible */
    }

    h1, .stMarkdown, .stButton, .stRadio, .stSuccess, .stText, .stFileUploader, .stTextInput, .stTextArea, .stMultiSelect {
        color: white !important;  /* Make all text white */
    }

    h1 {
        font-size: 3em;
        text-align: center;
        font-weight: bold;
        /* Remove box around title */
        background-color: transparent;
        padding: 10px;
        border-radius: 10px;
    }

    .stButton, .stRadio, .stFileUploader, .stTextInput, .stTextArea, .stMultiSelect {
        color: white !important;  /* Ensure buttons, file uploaders, etc., are also white */
        border: 2px solid white;
    }

    .stRadio input[type=radio], .stButton, .stFileUploader input[type="file"] {
        color: white !important;
    }

    /* Custom styling for info text (to make it black) */
    .stInfo {
        color: black !important;
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

# Reward animation
def reward_animation(predicted_digit):
    st.session_state.score += 1
    st.success(f"✅ **Predicted Digit:** `{predicted_digit}` 🔢")
    st.markdown('<div class="celebrate">🎉 Woohoo! Great job! 🎉</div>', unsafe_allow_html=True)

    # Simulate multiple balloons by repeating GIFs
    for _ in range(3):  # Adjust 1–5 for "amount of balloons"
        st.image("https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif", width=200)

# Title & Description
st.title("🧠 Handwritten Digit Recognizer")
st.markdown("🎯 **Recognize digits (0–9) drawn or uploaded by you!**")
st.markdown(f"🏆 **Score:** `{st.session_state.score}`")

# Reset button
if st.button("🔄 Reset Score"):
    st.session_state.score = 0
    st.info("🔁 Score has been reset!")

#
