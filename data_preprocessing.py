import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df, CONFIG):
    target = CONFIG["TARGET_COL"]
    numeric_cols = CONFIG["NUMERIC_COLS"]
    categorical_cols = CONFIG["CATEGORICAL_COLS"]

    # Fill missing values
    if "previous_year_rating" in df.columns:
        df["previous_year_rating"].fillna(0, inplace=True)
    if "education" in df.columns:
        df["education"].fillna(df["education"].mode()[0], inplace=True)

    # Encode categorical features
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Scale numeric features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X = df.drop(columns=[target])
    y = df[target]

    return X, y, df
