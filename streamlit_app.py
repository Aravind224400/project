import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os

# ----------------------------
# Load or train CNN model
# ----------------------------
@st.cache_resource  # Cache the model for faster loading
def load_or_train_model():
    model_path = "mnist_cnn_model_v2.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Build CNN model (potentially improved architecture)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    model.save(model_path)
    return model

model = load_or_train_model()

# ----------------------------
# Prediction Function
# ----------------------------
def predict_digit(image):
    # Add your prediction logic here
    # This is just a placeholder, replace with your actual code
    # For example:
    image = image.resize((28, 28))
    image = image.convert('L')  # Convert to grayscale
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model input
    
    prediction = model.predict(image_array)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return digit, confidence

# ----------------------------
# Streamlit UI
# ----------------------
st.title("Handwritten Digit Recognizer")

st.sidebar.header("About")
st.sidebar.markdown("This app recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    digit, confidence = predict_digit(image)
    st.success(f"Predicted Digit: **{digit}** (Confidence: {confidence:.2f}%)")

st.markdown("---")
st.markdown("**Note:** For best results, upload images of handwritten digits (0-9) that are clear and centered.")