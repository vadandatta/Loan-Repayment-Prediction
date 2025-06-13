import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df.drop(columns=['Member_ID', 'Loan_ID'], inplace=True)

    # Encode target
    df['Loan_repaid'] = df['Loan_repaid'].map({'Yes': 1, 'No': 0})

    # Encode categorical features (Gender and Loan_purpose)
    df = pd.get_dummies(df, columns=['Gender', 'Loan_purpose'], drop_first=True)

    X = df.drop('Loan_repaid', axis=1)
    y = df['Loan_repaid']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler
