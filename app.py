import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import time
import os
import pygame  # For sound effects

# Set page configuration
st.set_page_config(page_title="🧠 Handwritten Digit Recognizer", layout="centered")

# Initialize score and high score in session
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'high_score' not in st.session_state:
    st.session_state.high_score = 0

# Initialize pygame mixer for sound effects
pygame.mixer.init()

# Custom CSS for animations and styles
st.markdown("""
    <style>
    body {
        margin: 0;
        padding: 0;
    }

    .stApp {
        background: linear-gradient(270deg, #ff9a9e, #fad0c4, #fbc2eb, #a18cd1, #84fab0, #8fd3f4, #a6c1ee, #d4fc79);
        background-size: 1600% 1600%;
        animation: gradientShift 30s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
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

# Reward animation with sound effects
def reward_animation(predicted_digit):
    st.session_state.score += 1
    st.success(f"✅ **Predicted Digit:** `{predicted_digit}` 🔢")
    st.markdown('<div class="celebrate">🎉 Woohoo! Great job! 🎉</div>', unsafe_allow_html=True)

    # Play success sound
    pygame.mixer.music.load('success_sound.mp3')  # Make sure you have this file in your directory
    pygame.mixer.music.play()

    # Simulate multiple balloons by repeating GIFs
    for _ in range(3):  # Adjust 1–5 for "amount of balloons"
        st.image("https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif", width=200)

    # Update high score
    if st.session_state.score > st.session_state.high_score:
        st.session_state.high_score = st.session_state.score

# Timer challenge
def countdown_timer(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        remaining_time = int(duration - (time.time() - start_time))
        st.text(f"⏳ Time Left: {remaining_time} seconds")
        time.sleep(1)
        st.experimental_rerun()

# Title & Description
st.title("🧠 Handwritten Digit Recognizer")
st.markdown("🎯 **Recognize digits (0–9) drawn or uploaded by you!**")
st.markdown(f"🏆 **Score:** `{st.session_state.score}`")
st.markdown(f"🥇 **High Score:** `{st.session_state.high_score}`")

# Reset button
if st.button("🔄 Reset Score"):
    st.session_state.score = 0
    st.info("🔁 Score has been reset!")

# Choose input method
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
        key="canvas"
    )

    if st.button("🔍 Predict from Drawing"):
        countdown_timer(10)  # Set timer to 10 seconds

        if canvas_result.image_data is not None:
            img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
            processed = preprocess(img)
            prediction = model.predict(processed)
            predicted_digit = np.argmax(prediction)
            reward_animation(predicted_digit)

elif option == "📁 Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image of a digit (ideally 28x28 or larger)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", width=150)

        if st.button("🔍 Predict from Upload"):
            countdown_timer(10)  # Set timer to 10 seconds

            processed = preprocess(image)
            prediction = model.predict(processed)
            predicted_digit = np.argmax(prediction)
            reward_animation(predicted_digit)
