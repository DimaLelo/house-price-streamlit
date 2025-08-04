
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model, scaler, and feature names
model = joblib.load('house_price_model.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

# Set the title of the Streamlit app
st.title("🏡 House Price Predictor")

# User input fields for property features
square_feet = st.number_input("Square Feet", 500, 10000, 1500)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
lot_size = st.number_input("Lot Size", 500, 20000, 5000)
garage_size = st.slider("Garage Size", 0, 4, 1)
school_quality = st.slider("School Quality", 0.0, 10.0, 7.5)
distance_to_city_center = st.number_input("Distance to City Center (km)", 0.0, 100.0, 10.0)
property_tax_rate = st.number_input("Property Tax Rate (%)", 0.0, 10.0, 1.5)
median_neighborhood_income = st.number_input("Median Neighborhood Income ($)", 10000, 200000, 60000)
days_on_market = st.slider("Days on Market", 1, 365, 30)
year_built = st.number_input("Year Built", 1900, 2025, 2000)
location_type = st.selectbox("Location Type", ["rural", "suburban", "urban", "prestigious"])

# Feature engineering
current_year = 2025
property_age = current_year - year_built
proximity_score = 1 / (distance_to_city_center + 1)
total_rooms = bedrooms + bathrooms
expected_price_per_sqft = {'rural': 120, 'suburban': 180, 'urban': 300, 'prestigious': 450}[location_type]
is_luxury = int(expected_price_per_sqft > np.percentile([120, 180, 300, 450], 80))

# Create location features
location_features = {f"loc_{loc}": 0 for loc in ["prestigious", "rural", "suburban", "urban"]}
if f"loc_{location_type}" in location_features:
    location_features[f"loc_{location_type}"] = 1

# Prepare input data as a DataFrame
input_data = pd.DataFrame([{
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'lot_size': lot_size,
    'garage_size': garage_size,
    'school_quality': school_quality,
    'distance_to_city_center': distance_to_city_center,
    'property_tax_rate': property_tax_rate,
    'median_neighborhood_income': median_neighborhood_income,
    'days_on_market': days_on_market,
    'property_age': property_age,
    'proximity_score': proximity_score,
    'total_rooms': total_rooms,
    'expected_price_per_sqft': expected_price_per_sqft,
    'is_luxury': is_luxury,
    **location_features
}])

# Ensure all required features are present
for col in feature_names:
    if col not in input_data:
        input_data[col] = 0

input_data = input_data[feature_names]

# Scale numerical features
numerical_cols = ['square_feet', 'bedrooms', 'bathrooms', 'lot_size', 'garage_size',
                  'school_quality', 'distance_to_city_center', 'property_tax_rate',
                  'median_neighborhood_income', 'days_on_market', 'property_age',
                  'proximity_score', 'total_rooms', 'expected_price_per_sqft']

input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Predict price when the button is clicked
if st.button("Predict Price"):
    log_price = model.predict(input_data)[0]
    predicted_price = np.expm1(log_price)  # Inverse of log transformation
    st.success(f"🏠 Predicted House Price: **${predicted_price:,.0f}**")
