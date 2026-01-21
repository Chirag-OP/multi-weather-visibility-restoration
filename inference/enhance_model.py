import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

@st.cache_resource
def load_model():
    # 1. Load architecture
    generator = tf.keras.models.load_model("my_model", compile=False)

    # 2. Restore CHECKPOINT weights (THIS IS THE FIX)
    ckpt = tf.train.Checkpoint(generator1=generator)
    ckpt.restore(
        tf.train.latest_checkpoint("Code_Data/training_checkpoints/gen1")
    ).expect_partial()

    return generator

generator = load_model()

def preprocess_image(pil_image):
    image = np.array(pil_image.convert("RGB"))
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    image = tf.expand_dims(image, axis=0)
    return image

def postprocess_image(tensor):
    image = tensor[0].numpy()
    image = (image * 0.5 + 0.5) * 255.0
    image = np.clip(image, 0, 255).astype("uint8")
    return Image.fromarray(image)

def enhance_image(pil_image):
    inp = preprocess_image(pil_image)
    out = generator(inp, training=False)
    return postprocess_image(out)
