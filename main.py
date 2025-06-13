from src.preprocess import load_and_preprocess_data
from src.train_models import train_all_models
from src.evaluate_models import evaluate_best_model
from src.save_model import save_model

import os

# Ensure results and models directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load and preprocess data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("data/Micro-credit-Data-file.csv")

# Train all models
models = train_all_models(X_train, y_train, X_test, y_test)

# Evaluate and select best model
best_model = evaluate_best_model(models, X_test, y_test)

# Save best model and scaler
save_model(best_model, scaler)

print("Pipeline completed. Best model saved.")
