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
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #ef4444;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    div.stButton > button:first-child:hover {
        background-color: #dc2626;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Klasifikasi Jenis Mangrove")
st.write(
    "Upload gambar, lalu sistem akan memprediksi kelas mangrove menggunakan model MobileNetV3Large input 250x250."
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image_array = preprocess_image(uploaded_file)
    st.write("Gambar berhasil diupload.")

    if st.button("Prediksi", type="primary"):
        with st.spinner("Memproses prediksi..."):
            predictions = model.predict(image_array, verbose=0)[0]
            top3_idx = np.argsort(predictions)[-3:][::-1]
            top3 = [(CLASS_NAMES[i], float(predictions[i] * 100)) for i in top3_idx]

        pred_class, confidence = top3[0]

        st.success(f"Prediksi: {pred_class}")
        st.metric("Confidence", f"{confidence:.2f}%")

        st.subheader("Top-3 Prediksi")
        for i, (class_name, score) in enumerate(top3, start=1):
            st.write(f"{i}. {class_name}: {score:.2f}%")

if st.button("Reset"):
    st.rerun()
