import os
import numpy as np
import streamlit as st
import tensorflow as tf
import gdown
from PIL import Image

st.set_page_config(
    page_title="Klasifikasi Mangrove",
    page_icon="🌿",
    layout="wide"
)

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
    .main {
        padding-top: 1.2rem;
    }

    .title-box {
        background: linear-gradient(135deg, #0f766e, #14b8a6);
        padding: 22px 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(15, 118, 110, 0.18);
    }

    .title-box h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
    }

    .title-box p {
        margin: 8px 0 0 0;
        font-size: 1rem;
        opacity: 0.95;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    .panel-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        color: #111827;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.92rem;
        margin-top: 0.35rem;
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.25);
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        color: white;
    }

    .result-box {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 1rem;
    }

    .result-title {
        font-size: 0.95rem;
        color: #166534;
        margin-bottom: 0.35rem;
    }

    .result-class {
        font-size: 1.35rem;
        font-weight: 800;
        color: #14532d;
    }

    .top3-item {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }

    .footer-note {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-box">
        <h1>🌿 Klasifikasi Jenis Mangrove</h1>
        <p>
            Upload gambar daun atau objek mangrove, lalu sistem akan memprediksi kelas mangrove
            menggunakan model MobileNetV3Large dengan input 250×250.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

left_col, right_col = st.columns([1.05, 1], gap="large")

with left_col:
    st.markdown('<div class="panel-title">Input Gambar</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Pilih gambar",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="visible"
    )

    st.markdown(
        '<div class="small-note">Format yang didukung: JPG, JPEG, PNG, BMP, WEBP.</div>',
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.image(
            file_bytes,
            caption="Preview gambar input",
            width=320
        )
        st.markdown('</div>', unsafe_allow_html=True)

        uploaded_file.seek(0)
        image_array = preprocess_image(uploaded_file)

        predict_clicked = st.button("Prediksi", type="primary")
    else:
        predict_clicked = False
        st.info("Silakan upload gambar terlebih dahulu.")

with right_col:
    st.markdown('<div class="panel-title">Hasil Prediksi</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown(
            """
            <div class="panel">
                <p style="margin:0; color:#6b7280;">
                    Hasil klasifikasi akan tampil di sini setelah gambar diupload dan tombol prediksi ditekan.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif predict_clicked:
        with st.spinner("Memproses prediksi..."):
            predictions = model.predict(image_array, verbose=0)[0]
            top3_idx = np.argsort(predictions)[-3:][::-1]
            top3 = [(CLASS_NAMES[i], float(predictions[i] * 100)) for i in top3_idx]

        pred_class, confidence = top3[0]

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-title">Prediksi utama</div>
                <div class="result-class">{pred_class}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric("Confidence", f"{confidence:.2f}%")

        st.markdown("### Top-3 Prediksi")
        for i, (class_name, score) in enumerate(top3, start=1):
            st.markdown(
                f"""
                <div class="top3-item">
                    <b>{i}. {class_name}</b><br>
                    Probabilitas: {score:.2f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(min(int(confidence), 100))

    else:
        st.markdown(
            """
            <div class="panel">
                <p style="margin:0; color:#6b7280;">
                    Gambar sudah berhasil diupload. Tekan tombol <b>Prediksi</b> untuk melihat hasil klasifikasi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

col_reset, col_info = st.columns([1, 3])

with col_reset:
    if st.button("Reset"):
        st.rerun()

with col_info:
    st.markdown(
        '<div class="footer-note">Demo klasifikasi mangrove berbasis Streamlit · MobileNetV3Large</div>',
        unsafe_allow_html=True,
    )
