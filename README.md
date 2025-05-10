# Telecom Customer Churn Prediction

This application predicts whether a telecom customer will churn based on various customer attributes.

## Features

- Interactive web interface built with Streamlit
- Machine learning models trained on telecom customer data
- Visual churn probability indicators
- Customer insights and recommended actions

## How to Run Locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   streamlit run app.py
   ```

## Project Structure

- `app.py` - Streamlit web application
- `telecom_churn.py` - Model training script
- `TelcoCustomerChurn.csv` - Dataset
- `requirements.txt` - Dependencies
- `trained_model.pkl` - Saved model (generated after running training script)
- `scaler.pkl` - Feature scaler (generated after running training script)
