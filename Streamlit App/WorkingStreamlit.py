import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# Page configuration
st.set_page_config(page_title="Bank Account Prediction Dashboard", page_icon="💳")
st.title('Bank Account Prediction Dashboard')

# Load model and preprocessing objects
def load_model_objects():
    model_xgb = joblib.load('xgb_clf.joblib')
    scaler = joblib.load('scaler.joblib')
    encoder_y = joblib.load('encoder.joblib')  # For target variable
    le_country_economy = joblib.load('country_encoder.joblib')
    le_regionwb = joblib.load('regionwb_encoder.joblib')
    return model_xgb, scaler, encoder_y, le_country_economy, le_regionwb

model_xgb, _scaler, _label_encoder, le_country_economy, le_regionwb = load_model_objects()

@st.cache_data
def load_data():
    # Load the actual data from the CSV file
    return pd.read_csv(
        'micro_world_139countries.csv',
        encoding='ISO-8859-1'
    )

@st.cache_data
def process_data(df, _scaler, _label_encoder, _country_encoder, _regionwb_encoder):
    # Select relevant columns and sample
    sample_df = df[['remittances', 'educ', 'age', 'female', 'mobileowner',
                   'internetaccess', 'pay_utilities', 'receive_transfers',
                   'receive_pension', 'economy', 'regionwb', 'account']].sample(
                   n=5000, random_state=42, replace=True)
    
    # Drop rows with missing values in specified columns
    sample_df = sample_df.dropna(subset=['account', 'remittances', 'educ', 'age', 'female',
                                         'mobileowner', 'internetaccess', 'pay_utilities',
                                         'receive_transfers', 'receive_pension',
                                         'economy', 'regionwb']) 
    
    # Encode 'economy' using the loaded LabelEncoder
    sample_df['economy'] = _country_encoder.transform(sample_df['economy'])
    
    # Encode 'regionwb' using the loaded LabelEncoder
    sample_df['regionwb'] = _regionwb_encoder.transform(sample_df['regionwb'])
    
    # Manual encoding for 'educ'
    educ_mapping = {'None': 0, 'Primary': 1, 'Secondary': 2, 'Tertiary': 3}
    sample_df['educ'] = sample_df['educ'].map(educ_mapping).fillna(-1).astype(int)
    
    # Manual encoding for 'female'
    gender_mapping = {'Male': 0, 'Female': 1}
    sample_df['female'] = sample_df['female'].map(gender_mapping).fillna(-1).astype(int)
    
    # Convert boolean columns to integers
    boolean_columns = ['mobileowner', 'internetaccess', 'pay_utilities',
                       'receive_transfers', 'receive_pension']
    for col in boolean_columns:
        sample_df[col] = sample_df[col].astype(int)
    
    # Separate features and target
    X = sample_df.drop('account', axis=1)
    y = sample_df['account']
    
    # Encode target variable
    y = _label_encoder.transform(y)
    
    # Scale features using the loaded scaler
    X_scaled = _scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y

# Load data
df = load_data()
df = df.drop('inc_q', axis=1, errors='ignore')  # Ensure 'inc_q' is dropped if it exists

# Adding a sidebar for user input
with st.sidebar:
    st.title("Input User Data for Prediction")
    with st.form("user_inputs"):
        remittances = st.number_input('Remittances', min_value=0, max_value=100000, step=100)
        educ = st.selectbox('Education Level', options=['None', 'Primary', 'Secondary', 'Tertiary'])
        age = st.number_input('Age', min_value=18, max_value=100, step=1)
        female = st.selectbox('Gender', options=['Male', 'Female'])
        mobileowner = st.radio('Owns a Mobile', options=[True, False])
        internetaccess = st.radio('Has Internet Access', options=[True, False])
        pay_utilities = st.radio('Pays Utilities Online', options=[True, False])
        receive_transfers = st.radio('Receives Transfers', options=[True, False])
        receive_pension = st.radio('Receives Pension', options=[True, False])
        economy = st.selectbox('Country', options=list(le_country_economy.classes_))
        regionwb = st.selectbox('Region', options=list(le_regionwb.classes_))
        account = 1  # Placeholder or default value
        submit_button = st.form_submit_button("Predict")

# Processing user input for prediction
if submit_button:
    user_data = pd.DataFrame({
        'remittances': [remittances],
        'educ': [educ],
        'age': [age],
        'female': [female],
        'mobileowner': [mobileowner],
        'internetaccess': [internetaccess],
        'pay_utilities': [pay_utilities],
        'receive_transfers': [receive_transfers],
        'receive_pension': [receive_pension],
        'economy': [economy],
        'regionwb': [regionwb],
        'account': [account]
    })
    
    try:
        processed_user_data, _ = process_data(
            user_data, _scaler, _label_encoder, le_country_economy, le_regionwb
        )
        
        prediction = model_xgb.predict(processed_user_data)
        result = 'Has Bank Account' if prediction[0] == 1 else 'Does Not Have Bank Account'
        st.sidebar.write(f'Prediction: {result}')
    except Exception as e:
        st.sidebar.error(f"Error in processing data: {e}")

# Process example data
scaled_data, _ = process_data(df, _scaler, _label_encoder, le_country_economy, le_regionwb)

# Display the processed data in your Streamlit app
if scaled_data is not None:
    st.write("Scaled Data:", scaled_data)

# Main prediction logic
# Process the main dataset for predictions
processed_data, y_main = process_data(df, _scaler, _label_encoder, le_country_economy, le_regionwb)
if processed_data is not None:
    # Prepare features for prediction
    X = processed_data  # 'account' has been dropped in process_data
    y = y_main
    
    # Make predictions
    predictions = model_xgb.predict(X)

    # Show predictions
    st.write("Predictions:")
    st.write(predictions)

    # Plotting a confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, predictions)
    cm_fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(cm_fig)

    # Feature importance
    if st.button('Show Feature Importances'):
        feat_importances = pd.Series(model_xgb.feature_importances_, index=X.columns)
        st.bar_chart(feat_importances)