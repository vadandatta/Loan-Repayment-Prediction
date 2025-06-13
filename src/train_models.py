import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.ensemble import BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import Perceptron
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.metrics import accuracy_score

def train_all_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Ridge Classifier': RidgeClassifier(),
        'SGD Classifier': SGDClassifier(),
        'Passive Aggressive': PassiveAggressiveClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Extra Trees': ExtraTreesClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'SVC': SVC(),
        'Linear SVC': LinearSVC(),
        'NuSVC': NuSVC(),
        'KNN': KNeighborsClassifier(),
        'Radius Neighbors': RadiusNeighborsClassifier(),
        'Gaussian NB': GaussianNB(),
        'Bernoulli NB': BernoulliNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'MLP Classifier': MLPClassifier(max_iter=500),
        'Label Spreading': LabelSpreading(),
        'Label Propagation': LabelPropagation(),
        'Bagging': BaggingClassifier(),
        'Voting (Hard)': VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('svc', SVC())], voting='hard'),
        'Stacking': StackingClassifier(estimators=[('rf', RandomForestClassifier()), ('svc', SVC())], final_estimator=LogisticRegression()),
        'Gaussian Process': GaussianProcessClassifier(),
        'Calibrated CV': CalibratedClassifierCV(),
        'Nearest Centroid': NearestCentroid(),
        'Perceptron': Perceptron(),
        'Hist Gradient Boosting': HistGradientBoostingClassifier()
        # Add more if needed
    }

    results = []

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            results.append({'Model': name, 'Accuracy': accuracy})
        except Exception as e:
            results.append({'Model': name, 'Accuracy': None, 'Error': str(e)})

    results_df = pd.DataFrame(results)
    results_df.to_csv('results/model_comparison_results.csv', index=False)

    return models