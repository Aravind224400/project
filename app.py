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

# Custom CSS to set the provided aesthetic background image with higher opacity
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
        background: rgba(255, 255, 255, 0.9); /* Higher opacity (0.9) for a stronger overlay */
    }

    h1 {
        color: transparent;
        background: linear-gradient(to right, #f12711, #f5af19);
        -webkit-background-clip: text;
        font-size: 3em;
        text-align: center;
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
    image = image.resize((28, 28
