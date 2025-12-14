import streamlit as st
import pickle
import numpy as np

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_model():
    with open("car prediction.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Optional scaler
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

# ----------------------------------
# App UI
# ----------------------------------
st.set_page_config(page_title="Used Car Price Prediction", layout="centered")

st.title("🚗 Used Car Price Prediction App")
st.write("Enter car details to predict the price")

# ----------------------------------
# Input Fields (EDIT IF NEEDED)
# ----------------------------------
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2018)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
mileage = st.number_input("Mileage (km/l)", min_value=0.0, value=18.0)
engine = st.number_input("Engine (CC)", min_value=500, value=1200)
max_power = st.number_input("Max Power (bhp)", min_value=20.0, value=80.0)
seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("Predict Price"):
    input_data = np.array([[year, km_driven, mileage, engine, max_power, seats]])

    if scaler:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Car Price: ₹ {int(prediction[0]):,}")
