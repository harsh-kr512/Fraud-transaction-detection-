import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Function to load the pre-trained model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make predictions
def predict(model, features):
    # Predict whether the transaction is fraudulent (assuming your model outputs binary predictions)
    is_fraud = model.predict(features)
    return is_fraud

# Function to reverse label encoding
def reverse_label_encoding(index):
    transaction_types = ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    return transaction_types[index]

# Function to map boolean values to "True" and "False"
def map_boolean(value):
    return "True" if value else "False"

# Main function to run the Streamlit app
def main():
    # Set title and description
    st.title("Fraud Detection")
    st.write("This app detects fraudulent transactions.")
    
    # Load pre-trained model
    model_path = 'fraud.sav' 
    model = load_model(model_path)
    
    # Transaction types
    transaction_types = ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    
    # Get user input for transaction features
    st.sidebar.header("Enter Transaction Details")
    step = st.sidebar.number_input("Time taken(hr)")
    type_idx = st.sidebar.selectbox("Type of transaction", range(len(transaction_types)), format_func=lambda x: transaction_types[x])
    amount = st.sidebar.number_input("Amount")
    oldbalanceOrg = st.sidebar.number_input("Old Balance Origin")
    newbalanceOrig = st.sidebar.number_input("New Balance Origin")
    oldbalanceDest = st.sidebar.number_input("Old Balance Destination")
    newbalanceDest = st.sidebar.number_input("New Balance Destination")
    isFlaggedFraud = st.sidebar.selectbox("Is Flagged Fraud", ['False', 'True'])
    
    # Prepare features for prediction
    features = pd.DataFrame({
        'step': [step],
        'type': [type_idx],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'isFlaggedFraud': [isFlaggedFraud == 'True']
    })
    
    # Make prediction
    if st.sidebar.button("Detect Fraud"):
        is_fraud = predict(model, features)
        st.write("Is Fraudulent Transaction:", map_boolean(is_fraud))

    # Display selected transaction type in its original form
    st.write("Selected Transaction Type:", reverse_label_encoding(type_idx))

# Run the app
if __name__ == '__main__':
    main()

