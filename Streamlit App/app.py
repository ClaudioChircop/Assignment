import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import shap
from streamlit_shap import st_shap

# Page configuration
st.set_page_config(
    page_title="Airbnb Price Prediction in Copenhagen",
    page_icon="🏠")

st.title('Predict Airbnb Prices in Copenhagen')

# Display an image
st.image('https://gdkfiles.visitdenmark.com/files/382/164757_Nyhavn_Jacob-Schjrring-og-Simon-Lau.jpg?width=987', caption='Copenhagen', use_column_width=True)

# Load model and preprocessing objects
@st.cache_resource
def load_model_objects():
    model_xgb = joblib.load('model_xgb.joblib')
    scaler = joblib.load('scaler.joblib')
    ohe = joblib.load('ohe.joblib')
    return model_xgb, scaler, ohe

model_xgb, scaler, ohe = load_model_objects()

# Create SHAP explainer
explainer = shap.TreeExplainer(model_xgb)

# App description
with st.expander("What's this app?"):
    st.markdown("""
    This app helps you determine an appropriate nightly rate for your Airbnb listing in Copenhagen.
    We've trained an AI model on successful listings in Copenhagen to provide pricing suggestions based on a few key inputs.
    Our recommendation is to adjust the suggested price by about ±350 DKK depending on your specific amenities and the quality of your place.
    As a bonus feature 🌟, we've included an AI explainer 🤖 to help you understand the factors influencing the predicted price.
    """)

st.subheader('Describe your place')

# User inputs
col1, col2 = st.columns(2)

with col1:
    n_hood = st.selectbox('Neighborhood', options=ohe.categories_[0])
    room_type = st.radio('Room Type', options=ohe.categories_[1])
    instant_bookable = st.checkbox('Instant Booking Available')
    accommodates = st.number_input('Maximum Guests', min_value=1, max_value=16, value=2)

with col2:
    bedrooms = st.number_input('Number of Bedrooms', min_value=0, max_value=10, value=1)
    beds = st.number_input('Number of Beds', min_value=1, max_value=16, value=1)
    min_nights = st.number_input('Minimum Nights Stay', min_value=1, max_value=30, value=1)

# Prediction button
if st.button('Predict Price 🚀'):
    # Prepare categorical features
    cat_features = pd.DataFrame({'neighbourhood_cleansed': [n_hood], 'room_type': [room_type]})
    cat_encoded = pd.DataFrame(ohe.transform(cat_features).todense(), 
                               columns=ohe.get_feature_names_out(['neighbourhood_cleansed', 'room_type']))
    
    # Prepare numerical features
    num_features = pd.DataFrame({
        'instant_bookable': [instant_bookable],
        'accommodates': [accommodates],
        'bedrooms': [bedrooms],
        'beds': [beds],
        'minimum_nights_avg_ntm': [min_nights]
    })
    num_scaled = pd.DataFrame(scaler.transform(num_features), columns=num_features.columns)
    
    # Combine features
    features = pd.concat([num_scaled, cat_encoded], axis=1)
    
    # Make prediction
    predicted_price = model_xgb.predict(features)[0]
    
    # Display prediction
    st.metric(label="Predicted price per night", value=f'{round(predicted_price)} DKK')
    
    # Calculate and display price range
    lower_range = max(0, round(predicted_price - 350))
    upper_range = round(predicted_price + 350)
    st.write(f"Suggested price range: {lower_range} - {upper_range} DKK")
    
    # SHAP explanation
    st.subheader('Price Factors Explained 🤖')
    shap_values = explainer.shap_values(features)
    st_shap(shap.force_plot(explainer.expected_value, shap_values, features), height=400, width=600)
    
    st.markdown("""
    This plot shows how each feature contributes to the predicted price:
    - Blue bars push the price lower
    - Red bars push the price higher
    - The length of each bar indicates the strength of the feature's impact
    """)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and AI")