import argparse
import numpy as np
import joblib
import os
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

n_estimators = params["train"]["n_estimators"]
max_depth = params["train"]["max_depth"]

def main(config_path):
    # Thiết lập tracking URI cho MLflow
    mlflow.set_tracking_uri("file:///D:/Study/MLOps/Projects/mlruns")
    mlflow.set_experiment("Prediction")

    # Load dữ liệu huấn luyện
    data = np.load("data/processed/train.npz")
    X, y = data["X"], data["y"]

    # Load best hyperparameters từ Optuna (được lưu từ hyperparameter.py)
    best_params_path = "models/best_params.pkl"
    if not os.path.exists(best_params_path):
        raise FileNotFoundError("Best parameters not found. Hãy chạy scripts/hyperparameter.py trước.")

    best_params = joblib.load(best_params_path)

    # Huấn luyện mô hình
    model = RandomForestClassifier(**best_params)
    model.fit(X, y)

    # Dự đoán để log accuracy
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    # Tạo thư mục nếu chưa có
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("train_accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact("models/preprocessor.pkl")

    print("Mô hình đã huấn luyện và lưu tại models/model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    main(args.config)
