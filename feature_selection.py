import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def select_features(X, y, CONFIG):
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=5)
    rfe.fit(X, y)

    selected = X.columns[rfe.support_].tolist()
    return selected
