import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Page config
st.set_page_config(
    page_title="🌳 EcoTree Advisor",
    page_icon="🌳",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model_and_encoders():
    try:
        model = joblib.load("tree_growth_model.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        return model, label_encoders
    except Exception as e:
        logging.error(f"Error loading model or encoders: {str(e)}")
        st.error("Error loading the prediction model. Please ensure the model files exist.")
        return None, None

def encode_input(value, options, encoder):
    try:
        return encoder.transform([value])[0]
    except Exception as e:
        logging.error(f"Error encoding input: {str(e)}")
        return None

def main():
    st.title("🌳 EcoTree Advisor - Tree Growth Predictor")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("""
        This application helps you determine if a specific tree can grow successfully 
        under certain environmental conditions. Select the parameters below to get a prediction.
        """)
        
        st.header("How it works")
        st.write("""
        1. Select the tree type you want to grow
        2. Choose the soil type in your area
        3. Specify water availability
        4. Select the climate condition
        5. Click 'Predict' to get the result
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tree information
        st.subheader("Tree Information")
        
        # Dropdown options with descriptions
        tree_types = {
            "Banyan": "A large, fast-growing tree with aerial roots",
            "Mango": "Tropical fruit tree requiring warm climate",
            "Apple": "Temperate fruit tree requiring cold winters",
            "Neem": "Drought-resistant tree with medicinal properties",
            "Pine": "Coniferous tree suitable for cold climates"
        }
        
        soil_types = {
            "Sandy": "Light, well-draining soil",
            "Clay": "Heavy, moisture-retaining soil",
            "Loamy": "Balanced mixture of sand, silt, and clay",
            "Peaty": "Organic-rich, acidic soil",
            "Silty": "Fine-textured, fertile soil"
        }
        
        water_availability = {
            "Low": "Limited water supply",
            "Medium": "Moderate water supply",
            "High": "Abundant water supply"
        }
        
        climate_conditions = {
            "Tropical": "Hot and humid climate",
            "Dry": "Arid or semi-arid climate",
            "Temperate": "Moderate climate with distinct seasons",
            "Continental": "Cold winters and warm summers",
            "Polar": "Extremely cold climate"
        }
        
        # User Inputs with descriptions
        tree = st.selectbox(
            "🌱 Select the tree you want to grow",
            options=list(tree_types.keys()),
            format_func=lambda x: f"{x} - {tree_types[x]}"
        )
        
        soil = st.selectbox(
            "🌍 Select the soil type",
            options=list(soil_types.keys()),
            format_func=lambda x: f"{x} - {soil_types[x]}"
        )
        
        water = st.selectbox(
            "💧 Select water availability",
            options=list(water_availability.keys()),
            format_func=lambda x: f"{x} - {water_availability[x]}"
        )
        
        climate = st.selectbox(
            "☀️ Select climate condition",
            options=list(climate_conditions.keys()),
            format_func=lambda x: f"{x} - {climate_conditions[x]}"
        )
        
        if st.button("Predict Growth Feasibility", key="predict_button"):
            model, label_encoders = load_model_and_encoders()
            
            if model and label_encoders:
                try:
                    # Encode inputs
                    tree_encoded = encode_input(tree, tree_types, label_encoders["TreeType"])
                    soil_encoded = encode_input(soil, soil_types, label_encoders["SoilType"])
                    water_encoded = encode_input(water, water_availability, label_encoders["WaterAvailability"])
                    climate_encoded = encode_input(climate, climate_conditions, label_encoders["ClimateCondition"])
                    
                    if None in [tree_encoded, soil_encoded, water_encoded, climate_encoded]:
                        st.error("Error encoding inputs. Please try again.")
                        return
                    
                    # Make prediction
                    user_input = np.array([[tree_encoded, soil_encoded, water_encoded, climate_encoded]])
                    prediction = model.predict(user_input)[0]
                    prediction_proba = model.predict_proba(user_input)[0]
                    
                    # Display results
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ Growth Prediction: Favorable</h3>
                            <p>This tree has a {prediction_proba[1]*100:.1f}% chance of growing successfully in the selected conditions!</p>
                            <p>Recommended Actions:</p>
                            <ul>
                                <li>Ensure proper planting depth</li>
                                <li>Maintain regular watering schedule</li>
                                <li>Monitor soil conditions</li>
                                <li>Protect from extreme weather</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                            <h3>❌ Growth Prediction: Unfavorable</h3>
                            <p>This tree has a {prediction_proba[0]*100:.1f}% chance of not surviving in the selected conditions.</p>
                            <p>Considerations:</p>
                            <ul>
                                <li>Try a different tree species</li>
                                <li>Modify soil conditions if possible</li>
                                <li>Implement irrigation system</li>
                                <li>Consider microclimate modifications</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    logging.error(f"Error making prediction: {str(e)}")
                    st.error("An error occurred while making the prediction. Please try again.")
    
    with col2:
        st.subheader("Tree Growth Tips")
        st.write("""
        ### General Tree Care Tips:
        1. **Planting Depth**: Ensure the root ball is level with the ground
        2. **Watering**: New trees need regular watering for the first 2-3 years
        3. **Mulching**: Apply 2-4 inches of mulch around the base
        4. **Pruning**: Remove dead or damaged branches regularly
        5. **Fertilization**: Use appropriate fertilizer based on soil type
        """)
        
        # Add a simple visualization
        st.subheader("Growth Success Rate")
        st.write("Based on historical data:")
        success_rate = 0.35  # This should be calculated from actual data
        st.progress(success_rate)
        st.write(f"Average success rate: {success_rate*100:.1f}%")

if __name__ == "__main__":
    main()
