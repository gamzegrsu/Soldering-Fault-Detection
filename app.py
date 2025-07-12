import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Modeli yükle
model = load_model("lehim_hatasi_modeli.keras")

# Sınıf isimleri ve çözüm önerileri
class_names = ["soğuk_lehim", "eksik_lehim", "köprü", "aşırı_lehim", "lehım_çatlağı"]
cozum_onerileri = {
    "soğuk_lehim": "Lehimleme süresini artır. Havya ucunu temizle.",
    "eksik_lehim": "Lehim miktarını artır. Yüzey temizliğini kontrol et.",
    "köprü": "Aşırı lehimi temizle. Bileşenleri hizala.",
    "aşırı_lehim": "Daha az lehim kullan. Havya ile fazlalığı çek.",
    "lehım_çatlağı": "Termal stres kaynaklarını kontrol et. Yeniden lehimle."
}

# Streamlit Arayüzü
st.title("🔧 Lehimleme Hatası Tespiti")
st.markdown("Bir PCB görseli yükleyin, model hatayı tahmin etsin ve çözüm önerisi sunsun.")

uploaded_file = st.file_uploader("📤 Görsel yükle (.jpg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Görseli modele uygun hale getir
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    # Tahmin
    prediction = model.predict(input_tensor)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Sonuç
    st.markdown(f"### 🔍 Tahmin: `{predicted_class}` (%{confidence:.2f} güven)")
    st.markdown(f"### 💡 Öneri: {cozum_onerileri[predicted_class]}")
