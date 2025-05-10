import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# 1. Set page config (must be first!)
st.set_page_config(page_title="Hand Digit Recognizer", layout="centered")

# 2. Load model
model = load_model("mnist_cnn_model_v2.h5")

# 3. Title
st.title("Hand Digit Recognizer")

# 4. Tabs for drawing and upload
tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])

# -------- TAB 1: DRAWING -------- #
with tab1:
    st.subheader("Draw a digit (0-9)")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=12,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=192,
        height=192,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict from Drawing"):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            img = Image.fromarray(np.uint8(img)).convert("L")
            img = img.resize((28, 28))
            img = ImageOps.invert(img)
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            st.success(f"Predicted Digit: **{predicted_class}**")

# -------- TAB 2: UPLOAD -------- #
with tab2:
    st.subheader("Upload a digit image (28x28 grayscale or will be resized)")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict from Upload"):
            img = image.resize((28, 28))
            img = ImageOps.invert(img)
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            st.success(f"Predicted Digit: **{predicted_class}**")
