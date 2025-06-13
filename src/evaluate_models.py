import pandas as pd
from sklearn.metrics import classification_report


def evaluate_best_model(models, X_test, y_test):
    best_model_name = None
    best_accuracy = 0
    best_model = None

    for name, model in models.items():
        accuracy = model.score(X_test, y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)

    with open("results/updated_models_metrics.csv", "w") as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(report)

    return best_model
