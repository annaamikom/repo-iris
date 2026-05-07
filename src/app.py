# app.py
import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
# Load model
#import load model
from model_loader import load_model
#MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model_numpy.pkl")
#model = joblib.load(MODEL_PATH)
#model = joblib.load("model/model_numpy.pkl")
# Load model (cache biar ga reload terus)
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

class_names = ["Setosa", "Versicolor", "Virginica"]

# Halaman utama
st.set_page_config(page_title=" Prediksi Bunga Iris", layout="wide")
st.markdown("<h1 style='text-align: center; color: #6C63FF;'> Prediksi Bunga Iris dengan Machine Learning</h1>", unsafe_allow_html=True)
st.write("---")

# Sidebar
st.sidebar.header("Input Fitur")

sepal_length = st.sidebar.text_input("Panjang Sepal (cm)", "5.1")
sepal_width = st.sidebar.text_input("Lebar Sepal (cm)", "3.5")
petal_length = st.sidebar.text_input("Panjang Petal (cm)", "1.4")
petal_width = st.sidebar.text_input("Lebar Petal (cm)", "0.2")

try:
    inputs = pd.DataFrame([{
    "sepal_length": float(sepal_length),
    "sepal_width": float(sepal_width),
    "petal_length": float(petal_length),
    "petal_width": float(petal_width)
}])
    #input_data = np.array([inputs])
    prediction = model.predict(inputs)[0]
   
except ValueError:
    st.warning("⚠️ Masukkan semua input dengan format angka (misalnya: 5.1, 3.5, dst)")


# Layout 2 kolom
col1, col2 = st.columns(2)

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Contoh Bunga Iris",  use_container_width=True)

with col2:
    st.subheader("Hasil Prediksi")
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    st.success(f"Hasil prediksi model: **{class_names[prediction]}** ")

    st.markdown("---")
    st.markdown("**Fitur yang dimasukkan:**")
    st.json({
        "Sepal Length": sepal_length,
        "Sepal Width": sepal_width,
        "Petal Length": petal_length,
        "Petal Width": petal_width
    })

st.caption("Aplikasi ML dengan Streamlit & Scikit-learn ")