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

# Custom CSS to set the provided aesthetic background image with higher opacity (no overlay)
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

    h1 {
        color: white;  /* Making the title white */
        font-size: 3em;
        text-align: center;
        font-weight: bold;
    }

    .stButton button {
        color: white;  /* Button text color */
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
            processed = preprocess(image)
            prediction = model.predict(processed)
            predicted_digit = np.argmax(prediction)
            reward_animation(predicted_digit)
