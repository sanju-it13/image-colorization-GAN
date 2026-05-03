import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="Image Colorization",
    page_icon="🎨",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────
st.title("🎨 Automatic Image Colorization using GAN")
st.markdown("Upload a **grayscale image** and let the AI colorize it!")

# ── Load model ────────────────────────────────────────
@st.cache_resource
def load_model():
    import gdown
    import os
    
    if not os.path.exists('generator_final.keras'):
        with st.spinner('Downloading model... please wait'):
            gdown.download(
                'https://drive.google.com/uc?id=10qZoCgi1ZrZ0gG9haY25UR8akxAEP4p_',
                'generator_final.keras',
                quiet=False
            )

    model = tf.keras.models.load_model(
        'generator_final.keras',
        compile=False   # ✅ IMPORTANT
    )
    return model

generator = load_model()

# ── Helper functions ──────────────────────────────────
def preprocess(img):
    img = cv2.resize(img, (256, 256))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L   = (lab[:, :, 0:1] / 50.0) - 1.0
    return L

def postprocess(L_norm, ab_norm):
    L  = (L_norm  + 1.0) * 50.0
    ab =  ab_norm * 128.0 + 128.0
    lab = np.concatenate([L, ab], axis=-1)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb

def colorize(img_rgb):
    L = preprocess(img_rgb)
    ab_pred = generator(
        tf.constant(L[None], dtype=tf.float32),
        training=False
    )[0].numpy()
    colorized = postprocess(L, ab_pred)
    return colorized

# ── Upload section ────────────────────────────────────
st.markdown("---")
uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    # Load image
    img = Image.open(uploaded).convert('RGB')
    img_np = np.array(img)

    # Show upload
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Original Input")
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        st.image(gray, caption="Grayscale", use_container_width=True, clamp=True)

    # Colorize button
    with col2:
        st.subheader("🎨 Colorized Output")
        if st.button("✨ Colorize!", use_container_width=True):
            with st.spinner("Colorizing... please wait"):
                result = colorize(img_np)
                st.image(result, caption="GAN Colorized", use_container_width=True)

                st.success("Done! ✅")

                # Download button
                result_pil = Image.fromarray(result)
                import io
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Result",
                    data=buf.getvalue(),
                    file_name="colorized.png",
                    mime="image/png"
                )

    with col3:
        st.subheader("ℹ️ About")
        st.info("""
        **Model:** GAN (pix2pix)
        **Generator:** U-Net (7 blocks)
        **Discriminator:** PatchGAN
        **Dataset:** COCO 2017
        **Training:** 30 epochs
        **Resolution:** 256×256
        """)

# ── Footer ────────────────────────────────────────────
st.markdown("---")
st.markdown("Made using TensorFlow & Streamlit")