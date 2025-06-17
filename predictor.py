import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the models
with open("efficiency_model_improved.pkl", "rb") as f:
    efficiency_model = pickle.load(f)

with open("stability_material_rf_improved.pkl", "rb") as f:
    stability_model = pickle.load(f)

st.set_page_config(page_title="Perovskite Predictor", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
        body {
            background-color: #000;
            color: #fffd80;
        }
        h1, h2, h3, h4 {
            color: #fffd80 !important;
        }
        .stButton>button {
            background-color: #2980B9;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("☀️ Perovskite Solar Cell Stability & Efficiency Predictor")

st.markdown("""
### 🔬 What Does This Do?

Perovskite solar cells are a new generation of solar tech. These cells offer potential for high efficiency, low-cost production, and flexibility, making them an exciting area of research and development in solar energy. This tool predicts **efficiency** and **stability** using trained ML models.
""")

# ------------------ Stability Predictor ------------------
st.markdown("## 🌐 How This Website Helps Improve Stability?")
st.markdown("""
This platform leverages machine learning (ML) to analyze and predict the stability of perovskite materials under various environmental and fabrication conditions.
""")

st.markdown("### 🧪 Stability Predictor")
# Layout for inputs
col1, col2 = st.columns(2)
with col1:
    a_ion = st.selectbox("A-site ion", ["FA", "MA", "Cs", "Rb"])
    c_ion = st.selectbox("C-site ion", ["I", "Br", "Cl"])
    humidity = st.number_input("Synthesis Relative Humidity (%)", min_value=0.0, max_value=100.0, format="%.2f")
with col2:
    b_ion = st.selectbox("B-site ion", ["Pb", "Sn", "Ge"])
    bandgap = st.number_input("Bandgap (eV)", min_value=0.0, format="%.3f")

if st.button("Predict Stability"):
    input_df = pd.DataFrame({
        "A_site_material": [a_ion],
        "B_site_material": [b_ion],
        "C_site_material": [c_ion],
        "Perovskite_deposition_synthesis_atmosphere_relative_humidity": [humidity],
        "Perovskite_band_gap": [bandgap]
    })

    # ✅ Corrected interaction feature name
    input_df["Bandgap_Humidity_Interaction"] = input_df["Perovskite_band_gap"] * input_df["Perovskite_deposition_synthesis_atmosphere_relative_humidity"]

    prediction = stability_model.predict(input_df)
    st.success(f"Predicted Stability Score: {prediction[0]:.2f}")

# ------------------ Efficiency Predictor ------------------
st.markdown("---")
st.markdown("## ⚡ Efficiency Predictor")

st.markdown("""
Efficiency is the key performance metric for solar cells. Perovskite cells have reached high efficiency levels in labs, but real-world consistency remains a challenge.
""")

st.markdown("### 🔍 Estimate Efficiency")

# Layout for inputs
col1, col2 = st.columns(2)
with col1:
    thickness = st.number_input("Perovskite Thickness (nm)", min_value=0.0, format="%.1f")
    anneal_time = st.number_input("Annealing Time (minutes)", min_value=0.0, format="%.1f")
    htl = st.selectbox("HTL Material", ["Spiro-OMeTAD", "PTAA", "NiOx", "CuSCN"])
    htl_thick = st.number_input("HTL Thickness (nm)", min_value=0.0, format="%.1f")
with col2:
    temp = st.number_input("Deposition Temperature (°C)", min_value=0.0, format="%.1f")
    bandgap_eff = st.number_input("Bandgap1 (eV)", min_value=0.0, format="%.3f")
    etl = st.selectbox("ETL Material", ["TiO2", "SnO2", "ZnO", "PCBM"])
    etl_thick = st.number_input("ETL Thickness (nm)", min_value=0.0, format="%.1f")

if st.button("Predict Efficiency"):
    input_df = pd.DataFrame({
        "Perovskite_thickness_nm": [thickness],
        "Perovskite_deposition_temperature_C": [temp],
        "Perovskite_annealing_time_min": [anneal_time],
        "Perovskite_band_gap_eV": [bandgap_eff],
        "HTL_material": [htl],
        "HTL_thickness_nm": [htl_thick],
        "ETL_material": [etl],
        "ETL_thickness_nm": [etl_thick]
    })

    # ✅ Corrected interaction feature names
    input_df["Thickness_x_Annealing"] = input_df["Perovskite_thickness_nm"] * input_df["Perovskite_annealing_time_min"]
    input_df["Bandgap_x_Annealing"] = input_df["Perovskite_band_gap_eV"] * input_df["Perovskite_annealing_time_min"]

    prediction = efficiency_model.predict(input_df)[0]
    st.success(f"Predicted Efficiency: {prediction:.2f} %")
