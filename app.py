import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
st.set_page_config(
    layout="wide"
)
import io

def pil_to_bytes(pil_img, format="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    buf.seek(0)
    return buf
st.markdown("""
<style>
.c1 {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 100%;
}
.stButton > button {
    background-color: #8B5CF6;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #7C3AED;
    color: white;
}
</style>
""", unsafe_allow_html=True)
# from inference.enhancer import enhance_image
from inference.enhance_model import enhance_image
# from inference.enhancer import improver
input_image = None
input_np = None
enhanced_np = None
with st.sidebar:
    st.header(":violet[***Visibility Enhancer***]")
    # weather_type = st.selectbox("Weather Type", ["Fog", "Rain", "Haze"])
    # strength = st.slider("Enhancement Strength", 0.1, 1.0, 0.5)
    with st.expander("Upload Image"):
        uploaded_file=st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    enhance=st.button("Enhance", type="primary")
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        input_np = np.array(input_image)
    if enhance and uploaded_file is not None:
        with st.spinner("Enhancing Image..."):
            enhanced_np = enhance_image(input_image)
    elif enhance and uploaded_file is None:
        st.warning("Please upload an image first.")

c1= st.container()
with c1:
    c1.title("**Visibitilty Enhancer**" )
    c1.write("Enhances foggy and degraded images using deep learning")
col1,col2 = st.columns([1,1])
with col1:
    col1.write("Original Image")
    if uploaded_file:
        st.image(input_image, use_container_width=True, caption="Original Image")
with col2:
    st.write("Your output is")
    if enhanced_np is not None:
        # final_output=improver(enhanced_np,weather_type,strength)
        st.image(enhanced_np, use_container_width=True, caption="Enhanced Image")
        img_bytes = pil_to_bytes(enhanced_np)

        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=img_bytes,
            file_name="enhanced_image.png",
            mime="image/png"
        )