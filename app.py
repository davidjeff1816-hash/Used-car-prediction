import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    with open("car prediction.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Used Car Price Prediction", layout="centered")
st.title("🚗 Used Car Price Prediction")

# -----------------------------
# User Inputs
# -----------------------------
year = st.number_input("Year", 1990, 2025, 2018)
km_driven = st.number_input("Kilometers Driven", 0, 500000, 50000)
mileage = st.number_input("Mileage (km/l)", 0.0, 40.0, 18.0)
engine = st.number_input("Engine (CC)", 500, 5000, 1200)
max_power = st.number_input("Max Power (bhp)", 20.0, 400.0, 80.0)
seats = st.number_input("Seats", 2, 10, 5)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])

# -----------------------------
# Create input dataframe
# -----------------------------
input_df = pd.DataFrame({
    "year": [year],
    "km_driven": [km_driven],
    "mileage": [mileage],
    "engine": [engine],
    "max_power": [max_power],
    "seats": [seats],
    "fuel": [fuel],
    "seller_type": [seller_type],
    "transmission": [transmission],
    "owner": [owner]
})

# -----------------------------
# Encode categoricals
# (MUST match training)
# -----------------------------
input_df = pd.get_dummies(input_df)

# Align columns with model
model_features = model.feature_names_in_
input_df = input_df.reindex(columns=model_features, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"💰 Estimated Price: ₹ {int(prediction[0]):,}")
