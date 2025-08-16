from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models(X, y, selected_features, CONFIG):
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features], y, test_size=CONFIG["TEST_SIZE"], random_state=CONFIG["RANDOM_STATE"]
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=CONFIG["RANDOM_STATE"]),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=CONFIG["RANDOM_STATE"])
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models, X_train, X_test, y_train, y_test
