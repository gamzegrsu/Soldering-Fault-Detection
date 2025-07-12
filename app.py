import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

# Modeli yükle (aynı klasörde lehim_hatasi_modeli.keras olmalı)
model = load_model('lehim_hatasi_modeli.keras')

st.title("Lehim Hata Tespit ve Çözüm Önerisi")

# Sınıf isimleri (modeline göre değiştir)
classes = ['Soğuk Lehim', 'Eksik Lehim', 'Köprü', 'Aşırı Lehim', 'Lehim Çatlağı']

solutions = {
    'Soğuk Lehim': "Lehim sıcaklığını artırın ve lehimleme süresini optimize edin.",
    'Eksik Lehim': "Lehim miktarını artırın ve lehim pastasını eşit dağıtın.",
    'Köprü': "Lehim miktarını azaltın ve lehimleme sırasında bileşenleri sabitleyin.",
    'Aşırı Lehim': "Lehim miktarını azaltın ve sıcaklığı düşürün.",
    'Lehim Çatlağı': "Soğuma süresini yavaşlatın ve lehim kalitesini kontrol edin."
}

uploaded_file = st.file_uploader("Lehim görselinizi yükleyin", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # Model giriş boyutunu kontrol et
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    pred_idx = np.argmax(preds)
    pred_class = classes[pred_idx]
    confidence = preds[0][pred_idx]

    st.markdown(f"### Tahmin edilen hata: **{pred_class}**")
    st.markdown(f"### Güven skoru: **{confidence:.2f}**")
    st.markdown(f"### Çözüm Önerisi:")
    st.write(solutions[pred_class])
