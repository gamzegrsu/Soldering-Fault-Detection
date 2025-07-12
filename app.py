import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Modeli yükle
model = load_model('model/lehim_hatasi_modeli.keras')

# Kullanıcıdan görsel alma
uploaded_image = st.file_uploader("Lehim hatası resmi yükleyin", type=["jpg", "png", "jpeg"])

# Görseli gösterme
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Yüklenen Görsel', use_column_width=True)

    # Görseli uygun boyuta getirme
    image = image.resize((224, 224))  # Modelin istediği boyut
    image = np.array(image) / 255.0   # Normalize etme
    image = np.expand_dims(image, axis=0)  # Model için uygun şekle getirme

    # Tahmin yapma
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)

    # Sınıfları tanımlama (bunlar sizin modelinizin sınıflarına göre değişebilir)
    class_names = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7']
    st.write(f"Model Tahmini: {class_names[predicted_class[0]]}")
