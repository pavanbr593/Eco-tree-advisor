import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("tree_growth_model.pkl")

st.title("🌳 EcoTree Advisor - Tree Growth Predictor")

# Dropdown options
tree_types = ["Banyan", "Mango", "Apple", "Neem", "Pine"]
soil_types = ["Sandy", "Clay", "Loamy", "Peaty", "Silty"]
water_availability = ["Low", "Medium", "High"]
climate_conditions = ["Tropical", "Dry", "Temperate", "Continental", "Polar"]

# User Inputs
tree = st.selectbox("🌱 Select the tree you want to grow", tree_types)
soil = st.selectbox("🌍 Select the soil type", soil_types)
water = st.selectbox("💧 Select water availability", water_availability)
climate = st.selectbox("☀️ Select climate condition", climate_conditions)

# Encode inputs into numerical values (assuming a predefined mapping)
def encode_input(value, options):
    return options.index(value)

tree_encoded = encode_input(tree, tree_types)
soil_encoded = encode_input(soil, soil_types)
water_encoded = encode_input(water, water_availability)
climate_encoded = encode_input(climate, climate_conditions)

if st.button("Predict Growth Feasibility"):
    user_input = np.array([[tree_encoded, soil_encoded, water_encoded, climate_encoded]])
    prediction = model.predict(user_input)[0]
    
    if prediction == 1:
        st.success("✅ This tree can grow in the selected conditions!")
    else:
        st.error("❌ This tree might not survive in the selected conditions.")
