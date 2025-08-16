"""
HR Analytics - Employee Promotion Prediction
Author: Mohammad Karishma
GitHub: https://github.com/Karishma2324/HR-Analytics

Main script to run the ML pipeline:
1. Load & preprocess data
2. Feature selection
3. Train Logistic Regression, Random Forest, XGBoost
4. Evaluate & compare models
"""

import logging
import json
import os
import pandas as pd
from src.data_preprocessing import preprocess_data
from src.feature_selection import select_features
from src.model_training import train_models
from src.evaluation import evaluate_models, plot_roc_curves, plot_feature_importance

# -------------------------------
# CONFIG & LOGGING
# -------------------------------
with open("config.json") as f:
    CONFIG = json.load(f)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)

# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    logging.info("🚀 HR Analytics Pipeline Started")

    # Load dataset
    data_path = CONFIG["DATA_PATH"]
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}")
        return
    df = pd.read_csv(data_path)
    logging.info(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} cols")

    # Preprocess
    X, y, processed_df = preprocess_data(df, CONFIG)
    logging.info(f"✅ Data preprocessed, final shape: {X.shape}")

    # Feature Selection
    selected_features = select_features(processed_df, y, CONFIG)
    logging.info(f"✅ Selected features: {selected_features}")

    # Train Models
    models, X_train, X_test, y_train, y_test = train_models(X, y, selected_features, CONFIG)

    # Evaluate
    results = evaluate_models(models, X_test, y_test)
    logging.info("✅ Model evaluation complete")
    logging.info(results)

    # ROC & Feature Importance
    plot_roc_curves(models, X_test, y_test, save_path="figures/roc_curves.png")
    plot_feature_importance(models, selected_features, save_dir="figures/")

    logging.info("🎯 Pipeline finished successfully!")

if __name__ == "__main__":
    main()
