import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2
import os

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
# Prediction Function
# ----------------------------
def predict_digit_from_pil(image):
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28), Image.ANTIALIAS)
    image_array = np.array(image) / 255.0
    prediction = model.predict(image_array.reshape(1, 28, 28, 1))
    return int(np.argmax(prediction)), float(np.max(prediction)) * 100

def predict_digit_from_cv2(image_data):
    img = cv2.resize(image_data.astype('uint8'), (28, 28))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = gray / 255.0
    prediction = model.predict(gray.reshape(1, 28, 28, 1))
    return int(np.argmax(prediction)), float(np.max(prediction)) * 100

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🧠 Handwritten ✍🏻 Digit Recognizer")

st.sidebar.header("ℹ️ About")
st.sidebar.markdown("Recognize handwritten digits using a CNN trained on MNIST. Upload or draw a digit below.")

st.markdown("### 📥 Upload or ✏️ Draw a Digit")
tab1, tab2 = st.tabs(["Upload Image", "Draw Digit"])

# -------------
# Upload Tab
# -------------
with tab1:
    uploaded_file = st.file_uploader("Upload an image of a digit (0-9)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=False)
        digit, confidence = predict_digit_from_pil(image)
        if confidence < 60:
            st.warning(f"⚠️ Low confidence: {confidence:.2f}%. Try a clearer image.")
        else:
            st.success(f"✅ Predicted Digit: **{digit}** (Confidence: {confidence:.2f}%)")

# -------------
# Draw Tab
# -------------
with tab2:
    st.markdown("### ✏️ Draw a Digit (white on black):")

    canvas_result = st_canvas(
        fill_color="#ffffff",
        stroke_width=10,
        stroke_color="#ffffff",
        background_color="#000000",
        height=150,
        width=150,
        drawing_mode='freedraw',
        key="canvas2",
    )

    if canvas_result.image_data is not None:
        drawn_img = canvas_result.image_data
        img_resized = cv2.resize(drawn_img.astype('uint8'), (192, 192))
        st.image(img_resized, caption="Your Drawing", use_column_width=False)

        if st.button("Predict Drawn Digit"):
            digit, confidence = predict_digit_from_cv2(drawn_img)
            if confidence < 60:
                st.warning(f"⚠️ Low confidence: {confidence:.2f}%. Try redrawing.")
            else:
                st.success(f"✅ Predicted Digit: **{digit}** (Confidence: {confidence:.2f}%)")
                st.bar_chart(model.predict(np.expand_dims(cv2.cvtColor(cv2.resize(drawn_img.astype('uint8'), (28, 28)), cv2.COLOR_RGB2GRAY) / 255.0, axis=(0, -1)))[0])

st.markdown("---")
st.markdown("🔍 **Tip:** Draw large, centered digits or upload clear images similar to MNIST style.")
