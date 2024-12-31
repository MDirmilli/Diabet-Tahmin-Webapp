import streamlit as st
import joblib
import numpy as np

# En iyi modeli yükleme
model = joblib.load("eniyi.joblib")

# Kullanıcı girişi
st.title("Diabet Tahmini")
age = st.slider("Yaş", 20, 80, 30)
bmi = st.slider("BMI", 18.5, 35.0, 25.0)
blood_pressure = st.slider("Kan Basıncı", 60, 140, 80)
glucose = st.slider("Glikoz Seviyesi", 70, 200, 100)

# Tahmin yapma
if st.button("Tahmin Et"):
    input_data = np.array([[age, bmi, blood_pressure, glucose]])
    prediction = model.predict(input_data)
    st.write("Sonuç:", "Diabet" if prediction[0] == 1 else "Diabet Değil")
