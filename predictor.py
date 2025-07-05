import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the models
with open("efficiency_model_improved.pkl", "rb") as f:
    efficiency_model = pickle.load(f)

with open("stability_material_rf_enhanced.pkl", "rb") as f:
    stability_model = pickle.load(f)

st.set_page_config(page_title="Perovskite Predictor", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
        body {
            background-color: #F4F6F7;
            color: #007ACC;
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
    a_ion = st.selectbox("A-site ion", ["FA", "MA", "Cs"])
    c_ion = st.selectbox("C-site ion", ["I", "Br", "Cl"])
    humidity = st.number_input("Synthesis Relative Humidity (%)", min_value=0.0, max_value=100.0, format="%.2f")
    encapsulation = st.selectbox("Encapsulation", ["Yes", "No"])
    temp_stab = st.number_input("Temperature Stability (°C)", min_value=0.0, format="%.1f")
    defect_density = st.number_input("Defect Density (/cm³)", min_value=1e15, max_value=1e18, format="%.2e")
with col2:
    b_ion = st.selectbox("B-site ion", ["Pb", "Sn", "Ge"])
    a_bandgap = st.number_input("A-site Bandgap (eV)", min_value=0.0, format="%.2f")
    b_bandgap = st.number_input("B-site Bandgap (eV)", min_value=0.0, format="%.2f")
    c_bandgap = st.number_input("C-site Bandgap (eV)", min_value=0.0, format="%.2f")
    a_radius = st.number_input("A-site Ionic Radius (Å)", min_value=0.0, format="%.2f")
    b_radius = st.number_input("B-site Ionic Radius (Å)", min_value=0.0, format="%.2f")
    c_radius = st.number_input("C-site Ionic Radius (Å)", min_value=0.0, format="%.2f")
    a_affinity = st.number_input("A-site Electron Affinity (eV)", min_value=0.0, format="%.2f")
    b_affinity = st.number_input("B-site Electron Affinity (eV)", min_value=0.0, format="%.2f")
    c_affinity = st.number_input("C-site Electron Affinity (eV)", min_value=0.0, format="%.2f")

if st.button("Predict Stability"):
    input_df = pd.DataFrame({
        "A_site_material": [a_ion],
        "B_site_material": [b_ion],
        "C_site_material": [c_ion],
        "Humidity": [humidity],
        "Defect_density": [defect_density],
        "Encapsulation": [encapsulation],
        "Temperature_stability": [temp_stab],
        "A_radius": [a_radius],
        "B_radius": [b_radius],
        "C_radius": [c_radius],
        "A_electron_affinity": [a_affinity],
        "B_electron_affinity": [b_affinity],
        "C_electron_affinity": [c_affinity],
        "A_bandgap": [a_bandgap],
        "B_bandgap": [b_bandgap],
        "C_bandgap": [c_bandgap],
    })

    prediction = stability_model.predict(input_df)
    st.success(f"Predicted Stability: {prediction[0]:.1f} days")

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
    
