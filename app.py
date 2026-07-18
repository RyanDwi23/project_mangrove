import os
import numpy as np
import streamlit as st
import tensorflow as tf
import gdown
from PIL import Image

st.set_page_config(page_title="Klasifikasi Mangrove", layout="centered")

CLASS_NAMES = [
    "Avicennia Marina",
    "Avicennia Officinalis",
    "Avicennia Rumphiana",
    "Rhizophora Mucronata",
    "Sonneratia Alba",
]

IMAGE_SIZE = (250, 250)
MODEL_PATH = "bestmobilenetv3bs8lr1e5dropout05.keras"
FILE_ID = "1CUCILkto3mmD2KaRCFJFflZOqx73oL5b"


def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)


@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    display_img = img.copy()
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return display_img, img_array


st.title("Klasifikasi Jenis Mangrove")
st.write("Upload gambar, lalu sistem akan memprediksi kelas mangrove menggunakan model MobileNetV3Large input 250x250.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image, image_array = preprocess_image(uploaded_file)
    st.image(np.array(image), caption="Gambar input", use_container_width=True)

    if st.button("Prediksi", type="primary"):
        with st.spinner("Memproses prediksi..."):
            predictions = model.predict(image_array, verbose=0)[0]
            pred_idx = int(np.argmax(predictions))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(predictions[pred_idx] * 100)

        st.success(f"Prediksi: {pred_class}")
        st.metric("Confidence", f"{confidence:.2f}%")
