import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your trained models
@st.cache_data
def load_models():
    dt_model = joblib.load(open('decision_tree_model.pkl', 'rb'))
    lr_model = joblib.load(open('logistic_regression_model.pkl', 'rb'))
    svc_model = joblib.load(open('svm_model.pkl', 'rb'))
    rf_model = joblib.load(open('random_forest_model.pkl', 'rb'))  # Load Random Forest model
    return dt_model, lr_model, svc_model, rf_model

# Function to preprocess input data
def preprocess_input(input_data):
    # Create a DataFrame for the input data
    input_df = pd.DataFrame([input_data])

    # Encode categorical variables using LabelEncoder
    label_encoders = {}
    for column in input_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        input_df[column] = le.fit_transform(input_df[column])
        label_encoders[column] = le  # Store the encoder for future use

    return input_df

# Function to make predictions
def predict(model, input_data):
    input_data = preprocess_input(input_data)
    return model.predict(input_data)

def main():
    st.title("Risk Rating Prediction")

    # User input fields
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", options=["Male", "Female", "Non-binary"])
    education_level = st.selectbox("Education Level", options=["PhD", "Master's", "Bachelor's"])
    marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced", "Widowed"])
    income = st.number_input("Income")
    credit_score = st.number_input("Credit Score")
    loan_amount = st.number_input("Loan Amount")
    loan_purpose = st.selectbox("Loan Purpose", options=["Business", "Auto", "Home", "Personal"])
    employment_status = st.selectbox("Employment Status", options=["Employed", "Unemployed", "Self-Employed"])
    years_at_current_job = st.number_input("Years at Current Job")
    payment_history = st.selectbox("Payment History", options=["Excellent", "Good", "Fair", "Poor"])
    debt_to_income_ratio = st.number_input("Debt-to-Income Ratio")
    assets_value = st.number_input("Assets Value")
    number_of_dependents = st.number_input("Number of Dependents")
    city = st.text_input("City")
    state = st.text_input("State")
    country = st.text_input("Country")
    previous_defaults = st.number_input("Previous Defaults")
    marital_status_change = st.number_input("Marital Status Change")

    # Model selection
    model_option = st.selectbox("Select Model", options=["Decision Tree", "Logistic Regression", "SVM", "Random Forest"])  # Added Random Forest

    # When the user clicks the predict button
    if st.button("Predict"):
        # Prepare input data as a dictionary
        input_data = {
            'Age': age,
            'Gender': gender,
            'Education Level': education_level,
            'Marital Status': marital_status,
            'Income': income,
            'Credit Score': credit_score,
            'Loan Amount': loan_amount,
            'Loan Purpose': loan_purpose,
            'Employment Status': employment_status,
            'Years at Current Job': years_at_current_job,
            'Payment History': payment_history,
            'Debt-to-Income Ratio': debt_to_income_ratio,
            'Assets Value': assets_value,
            'Number of Dependents': number_of_dependents,
            'City': city,
            'State': state,
            'Country': country,
            'Previous Defaults': previous_defaults,
            'Marital Status Change': marital_status_change
        }

        # Load models
        dt_model, lr_model, svc_model, rf_model = load_models()

        # Predict based on selected model
        if model_option == "Decision Tree":
            prediction = predict(dt_model, input_data)
        elif model_option == "Logistic Regression":
            prediction = predict(lr_model, input_data)
        elif model_option == "SVM":
            prediction = predict(svc_model, input_data)
        else:  # For Random Forest
            prediction = predict(rf_model, input_data)

        st.success(f"The predicted Risk Rating is: {prediction[0]}")

if __name__ == "__main__":
    main()
