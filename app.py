import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Model ve sınıf bilgisi
model = load_model("lehim_hatasi_modeli.keras")
class_names = ["soğuk_lehim", "eksik_lehim", "köprü", "aşırı_lehim", "lehım_çatlağı"]
cozum_onerileri = {
    "soğuk_lehim": "Lehimleme süresini artır. Havya ucunu temizle.",
    "eksik_lehim": "Lehim miktarını artır. Yüzey temizliğini kontrol et.",
    "köprü": "Aşırı lehimi temizle. Bileşenleri hizala.",
    "aşırı_lehim": "Daha az lehim kullan. Havya ile fazlalığı çek.",
    "lehım_çatlağı": "Termal stres kaynaklarını kontrol et. Yeniden lehimle."
}

# Başlık
st.title("🔧 Lehimleme Hatası Tespiti ve Çözüm Önerisi")

# Görsel yükleme
uploaded_file = st.file_uploader("📤 Lütfen bir PCB görseli yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Görseli oku ve göster
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Yüklenen Görsel", use_column_width=True)

    # Görseli modele uygun boyuta getir
    resized = cv2.resize(image_rgb, (224, 224)) / 255.0
    input_data = np.expand_dims(resized, axis=0)

    # Tahmin yap
    prediction = model.predict(input_data)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Sonuçları yazdır
    st.markdown(f"### 🔍 Tahmin Edilen Hata: `{predicted_class}` ({confidence:.2f}%)")
    st.markdown(f"### 💡 Çözüm Önerisi:\n{cozum_onerileri[predicted_class]}")

