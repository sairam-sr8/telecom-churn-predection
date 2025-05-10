import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os

# Set page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📱",
    layout="wide"
)

# Title and description
st.title("Telecom Customer Churn Prediction")
st.write("""
This application predicts whether a telecom customer will churn based on various factors.
Enter the customer details below to get a prediction.
""")

# Check if models are already trained, otherwise train them
def load_or_train_models():
    model_file = 'trained_model.pkl'
    scaler_file = 'scaler.pkl'
    
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        # Load pre-trained model and scaler
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    else:
        # If models don't exist, run the training script
        st.warning("Training models first. This might take a minute...")
        import telecom_churn
        
        # After training, check again for the model files
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
        else:
            st.error("Could not train or load models. Please check your data and code.")
            return None, None

# Load or train models
with st.spinner("Loading models..."):
    model, scaler = load_or_train_models()

# Create input form with major factors only
st.header("Customer Information")
st.write("Please enter information about the key factors that influence customer churn.")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    # Contract is a major factor in churn
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], 
                          help="Contract type has a strong impact on churn rate")
    
    # Tenure is a major factor
    tenure = st.slider("Tenure (months)", 0, 72, 12, 
                     help="How long the customer has been with the company")
    
    # Internet service type
    internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"], 
                                  help="Type of internet service")
    
    # Payment method is important
    payment_method = st.selectbox("Payment Method", 
                               ["Electronic check", "Mailed check", 
                                "Bank transfer (automatic)", "Credit card (automatic)"],
                               help="How the customer pays their bill")

with col2:
    # Monthly charges impact churn
    monthly_charges = st.slider("Monthly Charges ($)", 20, 150, 65, 
                              help="Monthly bill amount")
    
    # Tech support is a differentiator
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], 
                              help="Whether the customer has tech support")
    
    # Senior citizen status
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], 
                               help="Whether the customer is a senior citizen")
    
    # Paperless billing
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], 
                                   help="Whether the customer has paperless billing")

# Calculate total charges based on tenure and monthly charges
total_charges = tenure * monthly_charges

# Set default values for other required fields that aren't shown
gender = "Male"  # Default value
partner = "No"   # Default value
dependents = "No" # Default value
phone_service = "Yes" # Default value
multiple_lines = "No" # Default value
online_security = "No" # Default value
online_backup = "No" # Default value
device_protection = "No" # Default value
streaming_tv = "No" # Default value
streaming_movies = "No" # Default value

# Feature engineering function
def preprocess_input(data):
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Convert categorical variables to numeric
    categorical_map = {
        "No": 0, "Yes": 1,
        "Male": 0, "Female": 1,
        "No phone service": 2, "No internet service": 2,
        "DSL": 1, "Fiber optic": 2,
        "Month-to-month": 0, "One year": 1, "Two year": 2,
        "Electronic check": 0, "Mailed check": 1, 
        "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
    }
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map(categorical_map)
    
    # Feature engineering similar to training script
    # Contract features
    df['ContractValue'] = df['MonthlyCharges'] * df['tenure']
    df['MonthlyChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    # Service features
    service_columns = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                      'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                      'StreamingTV', 'StreamingMovies']
    df['TotalServices'] = df[service_columns].sum(axis=1)
    
    # Customer features
    df['HasFamily'] = ((df['Partner'] == 1) | (df['Dependents'] == 1)).astype(int)
    df['IsSenior'] = df['SeniorCitizen'].astype(int)
    
    # Contract type
    contract_map = {0: 1, 1: 12, 2: 24}
    df['ContractMonths'] = df['Contract'].map(contract_map)
    
    return df

# Add explainer section
with st.expander("Why these factors?"): 
    st.write("""
    These factors were selected based on their importance in predicting customer churn:
    
    - **Contract Type**: Month-to-month contracts have much higher churn rates than longer-term contracts
    - **Tenure**: Newer customers are more likely to churn than long-term customers
    - **Monthly Charges**: Higher charges correlate with higher churn rates
    - **Internet Service**: Fiber optic service users tend to churn more than DSL users
    - **Tech Support**: Customers without tech support are more likely to leave
    - **Payment Method**: Electronic check users have higher churn rates
    - **Paperless Billing**: Customers with paperless billing tend to churn more
    - **Senior Citizen**: Senior citizens have different churn patterns than other customers
    """)

# Make prediction button
if st.button("Predict Churn Probability"):
    if model is not None:
        # Prepare input data
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Preprocess the input
        processed_data = preprocess_input(input_data)
        
        # Scale the features
        if scaler is not None:
            scaled_data = scaler.transform(processed_data)
            
            # Make prediction
            churn_prob = model.predict_proba(scaled_data)[0][1]
            churn_prediction = "Likely to Churn" if churn_prob > 0.5 else "Unlikely to Churn"
            
            # Display results
            st.header("Prediction Results")
            
            # Create columns for results
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric("Churn Probability", f"{churn_prob:.2%}")
            
            with res_col2:
                st.metric("Prediction", churn_prediction)
            
            # Add visualization
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 2))
            colors = ["green", "yellow", "red"]
            
            # Create a gradient from 0 to 1
            cmap = plt.cm.RdYlGn_r
            norm = plt.Normalize(0, 1)
            
            # Plot the gauge
            plt.barh(0, width=1, height=0.2, color="lightgrey")
            plt.barh(0, width=churn_prob, height=0.2, color=cmap(norm(churn_prob)))
            
            # Add markers
            for i, threshold in enumerate([0.25, 0.5, 0.75]):
                plt.axvline(x=threshold, color="grey", linestyle="--", alpha=0.7)
                plt.text(threshold, 0.3, f"{threshold:.0%}", ha="center")
            
            # Add pointer
            plt.scatter(churn_prob, 0, color="black", s=150, zorder=5)
            plt.text(churn_prob, -0.3, f"{churn_prob:.2%}", ha="center", fontweight="bold")
            
            # Configure the plot
            plt.xlim(0, 1)
            plt.ylim(-0.5, 0.5)
            plt.axis("off")
            plt.title("Churn Risk Meter", fontsize=14, pad=10)
            
            st.pyplot(fig)
            
            # Customer insights
            st.subheader("Customer Insights")
            if churn_prob > 0.7:
                st.error("This customer is at high risk of churning. Immediate retention actions recommended.")
                st.write("Possible actions:")
                st.write("- Offer a contract upgrade with pricing benefits")
                st.write("- Provide additional services at discounted rates")
                st.write("- Schedule a personal call to discuss customer satisfaction")
            elif churn_prob > 0.4:
                st.warning("This customer is at moderate risk of churning. Proactive retention may be beneficial.")
                st.write("Possible actions:")
                st.write("- Offer service add-ons at promotional rates")
                st.write("- Send personalized 'we value your business' communications")
                st.write("- Check for service usage patterns and optimize offerings")
            else:
                st.success("This customer has a low probability of churning. Regular engagement recommended.")
                st.write("Possible actions:")
                st.write("- Continue regular service quality monitoring")
                st.write("- Consider cross-selling additional services")
                st.write("- Include in general customer satisfaction surveys")
        else:
            st.error("Scaler not loaded properly.")
    else:
        st.error("Model not loaded properly. Please check the model file.")

# Add footer
st.markdown("---")
st.markdown("Telecom Customer Churn Prediction | Created with Streamlit")
