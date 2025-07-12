import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow sürümünü kontrol et
st.title("TensorFlow ve Streamlit Test Uygulaması")
st.write(f"Yüklü TensorFlow sürümü: {tf.__version__}")

# GPU'nun varlığını kontrol et
if tf.config.list_physical_devices('GPU'):
    st.write("GPU cihazı başarıyla algılandı!")
else:
    st.write("GPU cihazı bulunamadı, CPU kullanılıyor.")

# Basit bir TensorFlow modelini oluşturma
st.write("Basit bir TensorFlow modelini test ediyoruz...")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Kullanıcıdan bir sayı al
input_value = st.number_input("Bir sayı girin:", min_value=0, max_value=100, value=5)

# Modeli eğitmek için veri
x = np.array([[i] for i in range(100)])
y = np.array([2 * i + 1 for i in range(100)])

model.compile(optimizer='adam', loss='mse')

# Modeli eğit
if st.button("Modeli Eğit ve Tahmin Et"):
    model.fit(x, y, epochs=10, verbose=0)
    prediction = model.predict([[input_value]])[0][0]
    st.write(f"Model tahmini: {prediction:.2f}")

