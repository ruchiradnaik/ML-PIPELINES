import joblib
import pandas as pd

# Load the exported model bundle
bundle = joblib.load("Logistic Regression.joblib")

# Prepare new data (same structure as training data)
new_data = pd.DataFrame([{
    "team": "USA",
    "country": "United States",
    "year": 2016,
    "events": 20,
    "athletes": 15,
    "age": 24.5,
    "height": 180.2,
    "weight": 75.4,
    "prev_medals": 5.0,
    "prev_3_medals": 12.0
}])

X_new = new_data[bundle['feature_columns']]

# Apply the same preprocessing
for col, encoder in bundle['encoders'].items():
    if col in X_new.columns:
        X_new[col] = encoder.transform(X_new[col].astype(str))

# Scale the features
X_new_scaled = bundle['scaler'].transform(X_new)

# Make predictions
predictions = bundle['model'].predict(X_new_scaled)

# For classification, decode predictions back to original labels
if bundle['task_type'] == 'classification' and bundle['target_column'] in bundle['encoders']:
    predictions = bundle['encoders'][bundle['target_column']].inverse_transform(predictions)

print("Predictions:", predictions)
                