import numpy as np
import optuna
import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import yaml

# Load configuration
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

n_trials = config["tune"]["n_trials"]
est_range = config["tune"]["n_estimators_range"]
depth_range = config["tune"]["max_depth_range"]

def train_model(n_estimators, max_depth, X, y):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    return acc

def objective(trial):
    mlflow.set_tracking_uri("file:///D:/Study/MLOps/Projects/mlruns")
    mlflow.set_experiment("Telco Churn Prediction")

    # Load processed training data
    data = np.load("data/processed/train.npz")
    X, y = data["X"], data["y"]

    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", est_range[0], est_range[1])
    max_depth = trial.suggest_int("max_depth", depth_range[0], depth_range[1])

    with mlflow.start_run(nested=True):
        acc = train_model(n_estimators, max_depth, X, y)

        # Log params & metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("train_accuracy", acc)

    return acc

def main():
    # Load processed training data
    data = np.load("data/processed/train.npz")
    X, y = data["X"], data["y"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Save best params
    os.makedirs("models", exist_ok=True)
    joblib.dump(study.best_params, "models/best_params.pkl")

    # Train best model
    best_params = study.best_params
    model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"]
    )
    model.fit(X, y)
    joblib.dump(model, "models/best_model.pkl")

    print("Best hyperparameters saved to models/best_params.pkl")
    print(" Best model saved to models/best_model.pkl")
    print("Best trial:", best_params)

if __name__ == "__main__":
    main()
# This script uses Optuna to tune hyperparameters for a RandomForestClassifier.
# It logs the parameters and metrics to MLflow and saves the best model.