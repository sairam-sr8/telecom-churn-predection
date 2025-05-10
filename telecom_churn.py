# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.combine import SMOTEENN
import xgboost as xgb
import lightgbm as lgb
import pickle
import os

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return model, accuracy

print("Step 1: Loading and preprocessing data...")
df = pd.read_csv("TelcoCustomerChurn.csv")

print("\nStep 2: Basic preprocessing...")
# Convert TotalCharges to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])

print("\nStep 3: Feature Engineering...")
# Contract features
df['ContractValue'] = df['MonthlyCharges'] * df['tenure']
df['MonthlyChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

# Service features
service_columns = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                  'StreamingTV', 'StreamingMovies']
df['TotalServices'] = df[service_columns].apply(
    lambda x: sum([1 for i in x if i == 'Yes' or i == 'Fiber optic' or i == 'DSL']), axis=1)

# Customer features
df['HasFamily'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
df['IsSenior'] = df['SeniorCitizen'].astype(int)

# Contract type
contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
df['ContractMonths'] = df['Contract'].map(contract_map)

print("\nStep 4: Data preparation...")
# Drop customerID and convert categorical variables
df = df.drop('customerID', axis=1)
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Prepare data for modeling
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nStep 5: Handling class imbalance with SMOTEENN...")
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X_scaled, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print("\nStep 6: Initializing and training models...")
# Initialize models with optimized parameters
models = {
    'Logistic Regression': LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbosity=0  # Suppress XGBoost warnings
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=-1  # Suppress LightGBM warnings
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

# Train and evaluate individual models
model_results = {}
for name, model in models.items():
    trained_model, accuracy = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    model_results[name] = (trained_model, accuracy)

# Find the best model based on accuracy
best_model_name = max(model_results.items(), key=lambda x: x[1][1])[0]
best_model = model_results[best_model_name][0]
print(f"\nBest Model: {best_model_name} with accuracy: {model_results[best_model_name][1]:.4f}")

# Save the best model and scaler for deployment
print("\nSaving model and scaler for deployment...")
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")


