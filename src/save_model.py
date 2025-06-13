import joblib

def save_model(model, scaler):
    joblib.dump(model, 'models/final_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
