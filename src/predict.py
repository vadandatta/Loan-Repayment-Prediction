import joblib
import pandas as pd

def predict_new(input_csv):
    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    new_data = pd.read_csv(input_csv)
    new_data = pd.get_dummies(new_data, columns=['Gender', 'Loan_purpose'], drop_first=True)
    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    return predictions