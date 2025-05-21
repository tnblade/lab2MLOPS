import pandas as pd
import numpy as np
import os
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def clean_and_split(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = df.drop("customerID", axis=1)
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_cols),
        ('cat', categorical_pipeline, cat_cols)
    ])

    return preprocessor

def main(config_path):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Đọc dữ liệu...")
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    print("Tiền xử lý và tách train/test...")
    X_train, X_test, y_train, y_test = clean_and_split(df)

    print("Xây pipeline tiền xử lý...")
    preprocessor = build_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("Lưu dữ liệu đã xử lý...")
    np.savez_compressed("data/processed/train.npz", X=X_train_transformed, y=y_train)
    np.savez_compressed("data/processed/test.npz", X=X_test_transformed, y=y_test)

    joblib.dump(preprocessor, "models/preprocessor.pkl")

    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=preprocessor.get_feature_names_out())
    X_train_transformed_df["Churn"] = y_train.values
    X_train_transformed_df.to_csv("data/processed/train.csv", index=False)

    print("Hoàn tất xử lý dữ liệu. Dữ liệu đã lưu tại data/processed/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    main(args.config)
