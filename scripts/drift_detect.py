import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import joblib
import os
import pandas as pd

def main():
    print("Loading processed data and preprocessor...")
    
    if not os.path.exists("data/processed/train.npz") or not os.path.exists("data/processed/test.npz"):
        raise FileNotFoundError("Processed data files not found. Run `data_load.py` first.")

    if not os.path.exists("models/preprocessor.pkl"):
        raise FileNotFoundError("Preprocessor not found. Ensure preprocessing has run successfully.")

    # Load the training and testing data
    train = np.load("data/processed/train.npz")
    test = np.load("data/processed/test.npz")
    preprocessor = joblib.load("models/preprocessor.pkl")

    X_train = train["X"]
    X_test = test["X"]

    # Convert to pandas DataFrames with feature names
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    print("Running Data Drift Report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=X_train_df, current_data=X_test_df)

    os.makedirs("reports", exist_ok=True)
    report_html_path = "reports/drift_report.html"
    report_json_path = "reports/drift_report.json"
    report.save_html(report_html_path)
    report.save_json(report_json_path)

    print(f"Drift report saved at: {report_html_path} and {report_json_path}")

if __name__ == "__main__":
    main()