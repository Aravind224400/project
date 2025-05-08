import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os
from streamlit_drawable_canvas import st_canvas

# ----------------------------
# Load or Train CNN Model
# ----------------------------
@st.cache_resource
def load_or_train_model():
    model_path = "mnist_cnn_model_v3.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    model.save(model_path)
    return model

model = load_or_train_model()

# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(image):
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28), Image.ANTIALIAS)
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 28, 28, 1)

# ----------------------------
# Predict Digit
# ----------------------------
def predict_digit(image):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    return digit, confidence, prediction[0]

# ----------------------------
# Streamlit UI
# ----------------------------
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

# Upload Tab
with tab1:
    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)

        digit, confidence, probs = predict_digit(image)

        st.markdown(f"### ✅ Predicted: `{digit}`")
        st.progress(confidence / 100)
        st.caption(f"Confidence: {confidence:.2f}%")
        st.bar_chart(probs)

# Draw Tab
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

        if st.button("🔍 Predict"):
            digit, confidence, probs = predict_digit(drawn_image)

            st.markdown(f"### ✅ Predicted: `{digit}`")
            st.progress(confidence / 100)
            st.caption(f"Confidence: {confidence:.2f}%")
            st.bar_chart(probs)

st.markdown("---")
st.info("Tip: Draw clearly and center your digit. The model mimics the MNIST dataset.")
