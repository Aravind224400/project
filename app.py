import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Page configuration
st.set_page_config(page_title="Handwritten Digit Recognizer", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model_v2.h5")

model = load_model()

# Title
st.title("Handwritten Digit Recognizer")
st.markdown("Draw a digit or upload an image (0–9) and click **Predict**.")

# Tabs: Draw | Upload
tab1, tab2 = st.tabs(["✏️ Draw Digit", "📤 Upload Image"])

# ------------------------
# Image Preprocessing
# ------------------------
def preprocess_image(image):
    image = image.convert("L")                  # Convert to grayscale
    image = ImageOps.invert(image)              # Invert (white digit on black)
    image = ImageOps.autocontrast(image)        # Enhance contrast
    image = image.resize((28, 28))              # Resize to 28x28
    image_array = np.array(image) / 255.0       # Normalize
    return image_array.reshape(1, 28, 28, 1)     # Reshape for model

# ------------------------
# Prediction Function
# ------------------------
def predict_digit(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    return predicted_class, confidence, prediction[0]

# ------------------------
# Draw Tab
# ------------------------
with tab1:
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas_draw"
    )

    if st.button("Predict from Drawing"):
        if canvas_result.image_data is not None:
            image = Image.fromarray(np.uint8(canvas_result.image_data)).convert("RGB")
            digit, confidence, probs = predict_digit(image)
            st.success(f"Predicted Digit: **{digit}**")
            st.caption(f"Confidence: {confidence:.2f}%")
            st.bar_chart(probs)
        else:
            st.warning("Please draw a digit before clicking Predict.")

# ------------------------
# Upload Tab
# ------------------------
with tab2:
    uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=150)

        if st.button("Predict from Upload"):
            digit, confidence, probs = predict_digit(image)
            st.success(f"Predicted Digit: **{digit}**")
            st.caption(f"Confidence: {confidence:.2f}%")
            st.bar_chart(probs)
