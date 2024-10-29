import streamlit as st
import numpy as np
import pandas as pd
import joblib  
from tensorflow.keras.models import load_model

# Load the scaler and the trained model
scaler = joblib.load('App/crop_recommendation_scaler.joblib')
model = load_model('App/crop_recommendation_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define class names
class_names = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee',
    6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize',
    12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 16: 'orange',
    17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}

# Function to predict crops based on user input
def predict_crops(N, P, K, temperature, humidity, ph, rainfall):
    # Assuming these are the feature names you used during scaling
    feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    new_input = pd.DataFrame(np.array([[N, P, K, temperature, humidity, ph, rainfall]]), columns=feature_names)
    
    new_input_scaled = scaler.transform(new_input)
    pred_probs = model.predict(new_input_scaled)
    pred_crop = np.argmax(pred_probs)
    
    # Return crop index and probability
    return pred_crop, round(pred_probs[0][pred_crop] * 100, 2)

# Adding the CSS styles
st.markdown(f'<style>{open("App/styles.css").read()}</style>', unsafe_allow_html=True)
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# Title with enhanced background and styles
st.markdown("<div class='title-style'>🌾 Crop Advisor 🌱</div>", unsafe_allow_html=True)

# Instruction card
st.markdown(""" 
    <div class="instruction-card">
        <p class="instruction-title">Instructions:</p>
        <div class="instruction-text">
            1. Fill in the soil and climate information below.<br>
            2. Click "Predict Crop" to see which crop is best for your land.<br>
            3. Check the recommended crop and how likely it is to grow well.<br>
        </div>
    </div>
""", unsafe_allow_html=True)

# Input card
st.markdown(""" 
    <div class="input-card">
        <p class="input-title">🌱✨📊 Please Provide Soil and Climate Data 🌾🌍🔍</p>
        <div class="input-text">
            <i>Note:Please fill out all required inputs for the best results.</i>
        </div>
    </div>
""", unsafe_allow_html=True)

# Initialize session state for inputs if not already set
if 'N' not in st.session_state:
    st.session_state.N = None
if 'P' not in st.session_state:
    st.session_state.P = None
if 'K' not in st.session_state:
    st.session_state.K = None
if 'temperature' not in st.session_state:
    st.session_state.temperature = None
if 'humidity' not in st.session_state:
    st.session_state.humidity = None
if 'ph' not in st.session_state:
    st.session_state.ph = None
if 'rainfall' not in st.session_state:
    st.session_state.rainfall = None

# Arrange input fields in two columns inside the input card
col1, col2 = st.columns(2)

with col1:
    st.session_state.N = st.number_input('🍃 Nitrogen (N) content in soil', min_value=0, max_value=200, value=st.session_state.N) 
    st.session_state.P = st.number_input('🌾 Phosphorous (P) content in soil', min_value=0, max_value=200, value=st.session_state.P)
    st.session_state.K = st.number_input('🌱 Potassium (K) content in soil', min_value=0, max_value=200, value=st.session_state.K)
    st.session_state.temperature = st.number_input('🌡️ Temperature (°C)', min_value=0.0, max_value=50.0, value=st.session_state.temperature)

with col2:
    st.session_state.humidity = st.number_input('💧 Humidity (%)', min_value=0.0, max_value=100.0, value=st.session_state.humidity)
    st.session_state.ph = st.number_input('🧪 pH of soil', min_value=0.0, max_value=14.0, value=st.session_state.ph)
    st.session_state.rainfall = st.number_input('🌧️ Rainfall (mm)', min_value=0.0, max_value=500.0, value=st.session_state.rainfall)




# Container for buttons and result display
with st.container():
    # Create two columns in a single row for the buttons
    col3, col4, col5, col6 = st.columns(4)

    # Predict button in the first column
    with col3:
        predict_button = st.button("🌱 Predict Crop", key="predict_button", use_container_width=True)

    # Clear All button in the second column
    with col5:
        clear_button = st.button("❌ Clear All", key="clear_button",use_container_width=True)

    # Check if all fields are filled before predicting
    if predict_button:
        if None not in [st.session_state.get("N"), st.session_state.get("P"), st.session_state.get("K"),
                        st.session_state.get("temperature"), st.session_state.get("humidity"),
                        st.session_state.get("ph"), st.session_state.get("rainfall")]:
            
            # Call the prediction function and get result
            key, value = predict_crops(
                st.session_state.N, st.session_state.P, st.session_state.K,
                st.session_state.temperature, st.session_state.humidity,
                st.session_state.ph, st.session_state.rainfall
            )
            

            # Display the output result after the buttons
            st.markdown(f"""
                <div class="recommendation-container">
                    <p class="recommendation-title">🔍🌾 Optimal Crop for Your Land 🔍🌍</p>
                    <p class="crop-name">🌱{" "+class_names[key].capitalize()+" "}🌱</p>
                    <p class="probability-text">
                        {class_names[key].capitalize()} is the best choice for your land based on the information you provided.
                        With a <span class="probability-value">{value}%</span>chance means it is likely to grow well in your specific conditions.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Reset all input values in session state after prediction
            for field in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
                st.session_state[field] = None
        else:
            st.warning("Please fill all the fields!")

    # Clear button action
    if clear_button:
        # Clear the input values by resetting them in session state
        for field in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
            st.session_state[field] = None
