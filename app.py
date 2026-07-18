import numpy as np
import streamlit as st
import tensorflow as tf
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
MODEL_PATH = "best_mobilenetv3_bs8_lr1e5_dropout05.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


st.title("Klasifikasi Jenis Mangrove")
st.write(
    "Upload gambar dari file explorer, lalu sistem akan memprediksi kelas mangrove "
    "menggunakan model MobileNetV3Large input 250x250."
)

st.info(
    "Letakkan file model `.keras` di folder yang sama dengan `app.py` atau sesuaikan nilai `MODEL_PATH`."
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Pilih gambar", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image, image_array = preprocess_image(uploaded_file)
    st.image(image, caption="Gambar input", use_container_width=True)

    if st.button("Prediksi", type="primary"):
        with st.spinner("Memproses prediksi..."):
            predictions = model.predict(image_array, verbose=0)[0]
            pred_idx = int(np.argmax(predictions))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(predictions[pred_idx]) * 100

        st.success(f"Prediksi: {pred_class}")
        st.metric("Confidence", f"{confidence:.2f}%")

        st.subheader("Probabilitas per kelas")
        prob_data = {
            "Kelas": CLASS_NAMES,
            "Probabilitas": [float(p) * 100 for p in predictions],
        }
        st.bar_chart(prob_data, x="Kelas", y="Probabilitas")

        sorted_idx = np.argsort(predictions)[::-1]
        for i in sorted_idx:
            st.write(f"- {CLASS_NAMES[i]}: {predictions[i] * 100:.2f}%")
else:
    st.caption("Belum ada gambar yang di-upload.")
