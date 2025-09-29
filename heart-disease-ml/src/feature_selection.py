from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
import pandas as pd
import numpy as np


def rf_feature_importances(X, y, feature_names, n_top=10, random_state=42):
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X, y)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    top_features = [(feature_names[i], importances[i]) for i in idx[:n_top]]
    return top_features, importances, idx


def rfe_select(estimator, X, y, n_features_to_select=10):
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    return rfe


def chi2_select(X, y, k=10):
    # X must be non-negative for chi2. Scale if necessary or use SelectKBest with mutual_info_classif
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X, y)
    return selector