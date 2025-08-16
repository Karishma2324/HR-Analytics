import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
import numpy as np
import os

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = report
    return results

def plot_roc_curves(models, X_test, y_test, save_path="roc.png"):
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(models, features, save_dir="figures/"):
    os.makedirs(save_dir, exist_ok=True)
    if "XGBoost" in models:
        importances = models["XGBoost"].feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        plt.bar([features[i] for i in sorted_idx], importances[sorted_idx])
        plt.title("XGBoost Feature Importance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "xgboost_feature_importance.png"))
        plt.close()
