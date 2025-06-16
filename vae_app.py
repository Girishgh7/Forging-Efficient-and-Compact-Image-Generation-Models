import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Latent Space Image Generator", layout="centered")
st.title("Latent Space Image Generator using Decoder")

latent_dim = 128
num_images = st.slider("Number of images to generate", min_value=10, max_value=100, step=10, value=50)

@st.cache_resource
def load_decoder_model():
    return tf.keras.models.load_model("mountains_decoder.h5")

model = load_decoder_model()
fixed_noise = tf.random.normal((num_images, latent_dim))

with st.spinner("Generating images..."):
    y_pred = model.predict(fixed_noise)

def show_images(images, cols=10):
    rows = len(images) // cols + int(len(images) % cols != 0)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    for i in range(rows * cols):
        axes[i].axis('off')
        if i < len(images):
            img = images[i].squeeze()
            axes[i].imshow(img, cmap='gray')
    st.pyplot(fig)

show_images(y_pred)
