import time
import streamlit as st
from PIL import Image

from inference import predict
from image_quality import analyze_quality
from ela import generate_ela
from attention_map import create_heatmap

st.set_page_config(
    page_title="AI Deepfake Detector",
    layout="wide"
)

st.title("🧠 AI Deepfake Detector")
st.write("Vision Transformer (ViT) tabanlı Deepfake Tespit Sistemi")

uploaded_file = st.file_uploader(
    "Bir fotoğraf yükleyin",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Yüklenen Görsel",
        use_container_width=True
    )

    image.save("temp.jpg")

    if st.button("🔍 Analiz Et"):

        start_time = time.time()

        # -----------------------------
        # PREDICTION
        # -----------------------------
        result, mode, att = predict("temp.jpg")

        elapsed = time.time() - start_time

        st.subheader("📌 Tahmin Sonucu")

        if "FAKE" in result:
            st.error(result)
        else:
            st.success(result)

        st.info(f"⏱ Analiz Süresi: {elapsed:.2f} saniye")

        st.info(f"🧠 Analiz Modu: {mode}")

        # -----------------------------
        # IMAGE QUALITY
        # -----------------------------
        quality = analyze_quality("temp.jpg")

        if quality is not None:

            st.subheader("📊 Görüntü Kalite Analizi")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Çözünürlük",
                    f"{quality['width']} x {quality['height']}"
                )

            with col2:
                st.metric(
                    "Parlaklık",
                    f"{quality['brightness']}"
                )

            with col3:
                st.metric(
                    "Blur Skoru",
                    f"{quality['blur']}"
                )

            blur = quality["blur"]

            if blur < 30:
                st.error("⚠️ Görüntü çok bulanık, sonuç güvenilir olmayabilir.")
            elif blur < 100:
                st.warning("⚠️ Görüntüde bulanıklık var.")
            else:
                st.success("✅ Görüntü kalitesi uygun.")

        # -----------------------------
        # ELA ANALYSIS
        # -----------------------------
        st.subheader("🔍 Error Level Analysis (ELA)")

        ela_image = generate_ela("temp.jpg")

        st.image(
            ela_image,
            caption="ELA Görüntüsü",
            use_container_width=True
        )

        st.caption(
            "ELA görüntüsü, sıkıştırma ve düzenleme izlerini göstermek için kullanılır."
        )

        # -----------------------------
        # ATTENTION MAP
        # -----------------------------
        st.subheader("🧠 Attention Map (Model Nerelere Baktı?)")

        if att is not None:

            heatmap = create_heatmap("temp.jpg", att)

            st.image(
                heatmap,
                caption="ViT Attention Heatmap",
                use_container_width=True
            )