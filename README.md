# Loan Repayment Prediction in Microfinance using Machine Learning

This project develops and evaluates a machine learning-based solution to predict whether a borrower will repay a microloan within five days. The goal is to reduce default rates and improve loan decision-making in microfinance institutions (MFIs), using data-driven risk assessment.

---

##  Project Objective

- Predict the likelihood of timely loan repayment (within 5 days)
- Reduce credit default rates using predictive analytics
- Understand behavioral and financial patterns that influence repayment

---

##  Project Structure

```

Loan-Repayment-Prediction/
├── data/                            # Raw input dataset (download separately, see below)
├── models/                          # Saved final model and scaler (pkl files)
├── results/                         # Model performance summary and metrics
├── report/
│   └── Final\_Loan\_Repayment\_Report.pdf
├── src/                             # Source code modules
│   ├── preprocess.py
│   ├── train\_models.py
│   ├── evaluate\_models.py
│   ├── models.py
│   ├── predict.py
│   └── save\_model.py
├── main.py                          # Main script to execute pipeline
├── requirements.txt
├── README.md
└── .gitignore

````

---

##  Dataset Access

The dataset is not included in the GitHub repository due to file size.

To run the project, download the dataset from the link below and place it inside the data/ folder:

 Download CSV: https://drive.google.com/file/d/1DjXb-zLLybWlLP0JIZ67gUgyHBhFawcx/view?usp=sharing
File name: Micro-credit-Data-file.csv

---

##  Data Preprocessing

- Missing values dropped
- Categorical features one-hot encoded
- Numerical features scaled with StandardScaler
- Train-test split using stratified sampling

---

##  Exploratory Data Analysis (EDA)

- Gender and occupation impact repayment behavior
- Higher loan amounts reduce repayment probability
- Repayment varies by time of day and day of week

---

##  Feature Engineering

- Loan Amount to Age ratio
- Weekday extracted from timestamp
- IsWeekend flag

---

##  Model Selection

45+ models evaluated, including:

- Logistic Regression
- Decision Trees, Random Forest, XGBoost, LightGBM
- HistGradient Boosting
- Ensemble models (Voting, Bagging, Stacking)
- Naive Bayes, KNN, SVM, Neural Networks

---

##  Model Evaluation

Metrics:
- Accuracy
- Recall
- Log Loss
- Training Duration

Top Models:

| Model                 | Accuracy | Recall | Log Loss | Duration (s) |
|----------------------|----------|--------|----------|--------------|
| HistGradientBoosting | 93.35%   | 97.91% | 0.17988  | 2.78         |
| XGBoost              | 93.47%   | 97.75% | 0.10961  | 1.36         |
| LightGBM             | 92.83%   | 98.41% | 0.20700  | 1.67         |
| DecisionTree         | 92.28%   | 98.12% | 0.22010  | 1.76         |

Final Model: XGBoost (best balance of performance and efficiency)

---

##  Feature Importance

Top features from XGBoost:

- Loan Amount
- Transaction Time
- Device Type
- Usage Category
- Engineered Loan-to-Age Ratio

---

##  Business Applications

- Automate loan approvals
- Flag risky clients for manual review
- Reduce default rates and improve credit access
- Promote financial inclusion

---

##  How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ````

2. Download dataset and place it in the data/ folder.

3. Run the pipeline:

   ```bash
   python main.py
   ```

4. Outputs will be saved in models/ and results/ folders

---

##  Report

Refer to:
report/Final\_Loan\_Repayment\_Report.pdf

---

##  Author

* Vadan Datta

---

##  License

This project is for academic and educational purposes only. Please cite appropriately if reused.

```

