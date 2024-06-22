import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained model
model_filename = 'rand_for_model.joblib'

# Check if the model file exists
if not os.path.exists(model_filename):
    st.error(f"Model file '{model_filename}' not found. Please ensure it is in the correct directory.")
else:
    try:
        model = joblib.load(model_filename)
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False

# Define the preprocessor
categorical_cols = ['Credit_Mix']
numerical_cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 
                  'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 
                  'Outstanding_Debt', 'Credit_History_Age', 'Monthly_Balance']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(categories=[['Bad', 'Standard', 'Good']]), categorical_cols)
    ])

# Fit the preprocessor with some dummy data
dummy_data = pd.DataFrame({
    'Annual_Income': [50000], 'Monthly_Inhand_Salary': [4000],
    'Num_Bank_Accounts': [2], 'Num_Credit_Card': [1], 'Interest_Rate': [5],
    'Num_of_Loan': [1], 'Delay_from_due_date': [10], 'Num_of_Delayed_Payment': [5],
    'Outstanding_Debt': [10000], 'Credit_History_Age': [5], 'Monthly_Balance': [1000],
    'Credit_Mix': ['Good']
})
preprocessor.fit(dummy_data)

# Define a function for user input
def user_input_features():
    Annual_Income = st.number_input('Annual Income', min_value=0.0, value=50000.0)
    Monthly_Inhand_Salary = st.number_input('Monthly Inhand Salary', min_value=0.0, value=4000.0)
    Num_Bank_Accounts = st.number_input('Number of Bank Accounts', min_value=0, max_value=10, value=2)
    Num_Credit_Card = st.number_input('Number of Credit Cards', min_value=0, max_value=10, value=1)
    Interest_Rate = st.number_input('Interest Rate', min_value=0.0, max_value=100.0, value=5.0)
    Num_of_Loan = st.number_input('Number of Loans', min_value=0, max_value=10, value=1)
    Delay_from_due_date = st.number_input('Average number of days delayed by the person', min_value=0, max_value=100, value=10)
    Num_of_Delayed_Payment = st.number_input('Number of delayed payments', min_value=0, max_value=100, value=5)
    Credit_Mix = st.selectbox('Credit Mix', ['Bad', 'Standard', 'Good'])
    Outstanding_Debt = st.number_input('Outstanding Debt', min_value=0.0, value=10000.0)
    Credit_History_Age = st.number_input('Credit History Age', min_value=0, max_value=100, value=5)
    Monthly_Balance = st.number_input('Monthly Balance', min_value=0.0, value=1000.0)

    data = {'Annual_Income': Annual_Income,
            'Monthly_Inhand_Salary': Monthly_Inhand_Salary,
            'Num_Bank_Accounts': Num_Bank_Accounts,
            'Num_Credit_Card': Num_Credit_Card,
            'Interest_Rate': Interest_Rate,
            'Num_of_Loan': Num_of_Loan,
            'Delay_from_due_date': Delay_from_due_date,
            'Num_of_Delayed_Payment': Num_of_Delayed_Payment,
            'Outstanding_Debt': Outstanding_Debt,
            'Credit_History_Age': Credit_History_Age,
            'Monthly_Balance': Monthly_Balance,
            'Credit_Mix': Credit_Mix}

    features = pd.DataFrame(data, index=[0])
    return features

# Main function to run the app
def main():
    st.title("Credit Score Prediction App")
    st.write("This app predicts the credit score based on user input features.")
    
    # Get user input features
    input_df = user_input_features()

    if st.button('Predict'):
        # Check if model is loaded successfully
        if model_loaded:
            # Preprocess the input features
            input_preprocessed = preprocessor.transform(input_df)

            # Predict the credit score
            prediction = model.predict(input_preprocessed)
            prediction_proba = model.predict_proba(input_preprocessed)

            # Display the prediction
            st.subheader('Prediction')
            st.write(f'Predicted Credit Score: {prediction[0]}')
            
            st.subheader('Prediction Probability')
            st.write(prediction_proba)
        else:
            st.error("Model is not loaded. Please check the logs for details.")

if __name__ == '__main__':
    main()
