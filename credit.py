import streamlit as st
import numpy as np
import pickle
import gzip
import os


# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 20px;
        text-align: center;
        display: inline-block;
        font-size: 16px;
        margin: 10px 2px;
        cursor: pointer;
        border-radius: 10px;
        border: none;
    }
    .stTextInput > div > div > input {
        background-color: #e0e0e0;
        color: black;
    }
    .stNumberInput input {
        background-color: #e0e0e0;
        color: black;
    }
    .stSelectbox > div > div {
        background-color: #e0e0e0;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# Load your trained model
model_filename = 'rand_for_model.pkl.gz'

# Check if the model file exists
if not os.path.exists(model_filename):
    st.error(f"Model file '{model_filename}' not found. Please ensure it is in the correct directory.")
else:
    try:
        with gzip.open(model_filename, 'rb') as file:
            model = pickle.load(file)
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False

st.title("Credit Score Prediction")

# Create a form for user input
with st.form(key='credit_score_form'):
    annual_income = st.number_input("Annual Income:", min_value=0.0)
    monthly_inhand_salary = st.number_input("Monthly Inhand Salary:", min_value=0.0)
    num_bank_accounts = st.number_input("Number of Bank Accounts:", min_value=0)
    num_credit_cards = st.number_input("Number of Credit cards:", min_value=0)
    interest_rate = st.number_input("Interest rate:", min_value=0.0)
    num_loans = st.number_input("Number of Loans:", min_value=0)
    avg_days_delayed = st.number_input("Average number of days delayed by the person:", min_value=0)
    num_delayed_payments = st.number_input("Number of delayed payments:", min_value=0)
    credit_mix = st.selectbox("Credit Mix", options=["Good", "Standard", "Bad"])
    outstanding_debt = st.number_input("Outstanding Debt:", min_value=0.0)
    credit_history_age = st.number_input("Credit History Age:", min_value=0.0)
    monthly_balance = st.number_input("Monthly Balance:", min_value=0.0)

    submit_button = st.form_submit_button(label='Predict Credit Score')

# Convert credit mix to numerical value
credit_mix_dict = {"Bad": 0, "Standard": 1, "Good": 2}
credit_mix_value = credit_mix_dict[credit_mix]

# Create the features array
features = np.array([[annual_income, monthly_inhand_salary, num_bank_accounts, num_credit_cards, 
                      interest_rate, num_loans, avg_days_delayed, num_delayed_payments, 
                      credit_mix_value, outstanding_debt, credit_history_age, monthly_balance]])

# Predict and display the result
if submit_button:
    if model_loaded:
        result = model.predict(features)
        st.write(f"Predicted Credit Score = {result[0]}")
        
        if result == 2:
            st.success("Credit score: Good")
        elif result == 0:
            st.error("Credit score: Poor")
        else:
            st.warning("Credit score: Standard")
    else:
        st.error("Model could not be loaded. Please check the error message above.")

# To run the streamlit app, save this file as app.py and run the following command:
# streamlit run app.py
