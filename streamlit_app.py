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
        model = tf.keras.models.load_model(model_path)
        return model

    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Build improved CNN model
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
# Preprocessing Function
# ----------------------------
def preprocess_image(image):
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28), Image.ANTIALIAS)
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 28, 28, 1)

# ----------------------------
# Prediction Function
# ----------------------------
def predict_digit(image):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    return digit, confidence

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🧠 Handwritten Digit Recognizer")

st.sidebar.header("ℹ️ About")
st.sidebar.markdown("This app uses a Convolutional Neural Network (CNN) trained on MNIST to recognize handwritten digits. You can either upload an image or draw a digit below.")

st.markdown("### 📥 Upload or ✏️ Draw a Digit")

tab1, tab2 = st.tabs(["Upload Image", "Draw Digit"])

# Upload image tab
with tab1:
    uploaded_file = st.file_uploader("Upload an image of a digit (0-9)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=False)

        digit, confidence = predict_digit(image)

        if confidence < 60:
            st.warning(f"⚠️ Low confidence: {confidence:.2f}%. Try a clearer image.")
        else:
            st.success(f"✅ Predicted Digit: **{digit}** (Confidence: {confidence:.2f}%)")

# Draw digit tab
with tab2:
    st.write("Draw a digit below (white background, black digit):")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        drawn_image = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype("uint8"))
        st.image(drawn_image.resize((100, 100)), caption="Your Drawing", use_column_width=False)

        if st.button("Predict Drawn Digit"):
            digit, confidence = predict_digit(drawn_image)

            if confidence < 60:
                st.warning(f"⚠️ Low confidence: {confidence:.2f}%. Try redrawing.")
            else:
                st.success(f"✅ Predicted Digit: **{digit}** (Confidence: {confidence:.2f}%)")

st.markdown("---")
st.markdown("🔍 **Tip:** Draw large, centered digits or upload a clear image. The model expects digits similar in style to MNIST.")
