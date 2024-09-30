import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
import random
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
 
# Page configuration
st.set_page_config(page_title="Bank Account Prediction Dashboard", page_icon="💳")
st.title('Bank Account Prediction Dashboard')
 
# Load model and preprocessing objects
def load_model_objects():
    model_xgb = joblib.load('model_xgb.joblib')
    scaler = joblib.load('scaler.joblib')
    encoder = joblib.load('encoder.joblib')
    regionwb_encoder = joblib.load('regionwb_encoder.joblib')
    country_encoder = joblib.load('country_encoder.joblib')
    
    return model_xgb, scaler, encoder, regionwb_encoder, country_encoder
 
model_xgb, _scaler, _label_encoder,region_encoder, country_encoder = load_model_objects()
 
@st.cache
def load_data():
    # Load the actual data from the CSV file
    return pd.read_csv('micro_world_139countries.csv', encoding='ISO-8859-1')

@st.cache
def process_data(df, _scaler, _label_encoder,_region_encoder, _country_encoder):
    
    #print(df)
    sample_df = df[['remittances', 'educ', 'age', 'female', 'mobileowner','internetaccess', 'pay_utilities', 'receive_transfers','receive_pension', 'economy', 'regionwb','account']].sample(n=5000, random_state=42, replace = True)
    #print(sample_df.columns)
    
    cols=list(sample_df.columns)
    cols.remove('account')
    #print(cols)
    sample_df = sample_df.dropna(subset=['account','remittances', 'educ', 'age', 'female', 'mobileowner','internetaccess', 'pay_utilities', 'receive_transfers','receive_pension', 'economy', 'regionwb']) 
    
    
    
    sample_df['economy'] = country_encoder.fit_transform(sample_df['economy'])#Giving unique int values to economies
    sample_df['regionwb'] = region_encoder.fit_transform(sample_df['regionwb'])#Unique int values to regions
    X = sample_df.drop('account', axis=1)
    y = sample_df['account']
    y= _label_encoder.fit_transform(y)
    y=pd.DataFrame(y,columns=['account'])
    print(y.columns)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)
    X.columns = cols
    #X['account'] = y['account']
    return X
#def process_data(df, _scaler, _label_encoder, is_bulk_data=True):
    
    
    # Define label encoders outside the function to avoid re-initializing
    #if 'economy' in df.columns:
     #   df['economy'] = country_encoder.fit_transform(df['economy'])
    #if 'regionwb' in df.columns:
    #    df['regionwb'] = regionwb_encoder.fit_transform(df['regionwb'])
    # Ensure all required columns exist, except 'account' which might not be in user inputs
   # required_columns = ['inc_q', 'remittances', 'educ', 'age', 'female', 'mobileowner','internetaccess', 'pay_utilities', 'receive_transfers','receive_pension', 'economy', 'regionwb','account']
    #if is_bulk_data:
        #required_columns.append('account')
 
    #missing_cols = [col for col in required_columns if col not in df.columns]
    #if missing_cols:
     #   st.error(f"Missing columns: {', '.join(missing_cols)}")
      #  return None, None
 
    # Apply Label Encoding only if the column exists
   
    # Scale numerical features
   # scaled_features = _scaler.transform(df[required_columns[:-1]])  # Exclude 'account' for scaling if bulk data
    #if is_bulk_data:
        # Encode the target variable
     #   df['account_encoded'] = _label_encoder.transform(df['account'])
      #  return pd.DataFrame(scaled_features, columns=required_columns[:-1]), df['account_encoded']
    #else:
     #   return pd.DataFrame(scaled_features, columns=required_columns), None
 # Load data
df = load_data()
df.drop('inc_q',axis=1)
# Adding a sidebar for user input
with st.sidebar:
    st.title("Input User Data for Prediction")
    with st.form("user_inputs"):
        #inc_q = st.selectbox('Income Quartile', options=['Q1', 'Q2', 'Q3', 'Q4'])
        remittances = st.number_input('Remittances', min_value=0, max_value=100000, step=100)
        educ = st.selectbox('Education Level', options=['None', 'Primary', 'Secondary', 'Tertiary'])
        age = st.number_input('Age', min_value=18, max_value=100, step=1)
        female = st.radio('Gender', options=['Male', 'Female'])
        mobileowner = st.radio('Owns a Mobile', options=[True, False])
        internetaccess = st.radio('Has Internet Access', options=[True, False])
        pay_utilities = st.radio('Pays Utilities Online', options=[True, False])
        receive_transfers = st.radio('Receives Transfers', options=[True, False])
        receive_pension = st.radio('Receives Pension', options=[True, False])
        economy = st.selectbox('Country', options= list(df['economy'].unique()))
        regionwb = st.selectbox('Region', options=['South Asia', 'Europe & Central Asia (excluding high income)', 'Middle East & North Africa (excluding high income)',
 'Latin America & Caribbean (excluding high income)', 'High income',
 'Sub-Saharan Africa (excluding high income)',
 'East Asia & Pacific (excluding high income)'])
        account= 1
        submit_button = st.form_submit_button("Predict")
        
        
# Processing user input for prediction
if submit_button:
    user_data = pd.DataFrame({
        #'inc_q': [inc_q], 
        'remittances': [remittances], 'educ': [educ], 'age': [age], 'female': [female],
        'mobileowner': [mobileowner], 'internetaccess': [internetaccess], 'pay_utilities': [pay_utilities],
        'receive_transfers': [receive_transfers], 'receive_pension': [receive_pension], 'economy': [economy], 
        'regionwb': [regionwb], 'account' :[account]
    })
    
    processed_user_data = process_data(user_data, _scaler, _label_encoder,region_encoder, country_encoder)
    if processed_user_data is not None:
       #####
        #X = user_data.drop('account', axis=1)
        #y = user_data['account']
       #### 
        prediction = model_xgb.predict(processed_user_data)
        result = 'Has Bank Account' if prediction[0] == 1 else 'Does Not Have Bank Account'
        st.sidebar.write(f'Prediction: {result}')
 

 
# Process example data
scaled_data = process_data(df, _scaler, _label_encoder,region_encoder, country_encoder)
 
# Display the processed data in your Streamlit app
if scaled_data is not None:
    st.write("Scaled Data:", scaled_data)
    
 
# Main prediction logic
# Process the main dataset for predictions
processed_data = process_data(df, _scaler, _label_encoder,region_encoder, country_encoder)
if processed_data is not None:
    # Prepare features for prediction
    X = processed_data.drop('account',axis=1)  # Adjust based on your requirements
    y = processed_data['account']
    
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
        
