import pandas as pd
import joblib
import os
import argparse
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def save_report(name, acc, report_dict, matrix, y_true, y_pred):
    os.makedirs("reports", exist_ok=True)

    with open(f"reports/{name}_eval_report.json", "w") as f:
        json.dump(report_dict, f, indent=4)

    with open(f"reports/{name}_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(classification_report(y_true, y_pred))

    with open(f"reports/{name}_confusion_matrix.txt", "w") as f:
        f.write(str(matrix))

def evaluate_model(model_path, X, y, name):
    print(f"Evaluating: {name}")
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    matrix = confusion_matrix(y, y_pred)

    save_report(name, acc, report, matrix, y, y_pred)
    return acc

def main(config_path):
    print("Starting evaluation...")

    if not os.path.exists("data/processed/train.csv"):
        print(" Error: data/processed/train.csv not found.")
        return

    df = pd.read_csv("data/processed/train.csv")
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    acc_default = None
    acc_best = None

    # Evaluate default model
    if os.path.exists("models/model.pkl"):
        acc_default = evaluate_model("models/model.pkl", X, y, "default_model")
    else:
        print(" models/model.pkl not found.")

    # Evaluate best model if exists
    if os.path.exists("models/best_model.pkl"):
        acc_best = evaluate_model("models/best_model.pkl", X, y, "best_model")
    else:
        print(" models/best_model.pkl not found.")

    # Optionally: Save summary to compare models
    summary = {
        "default_model_accuracy": acc_default,
        "best_model_accuracy": acc_best
    }

    with open("reports/model_comparison.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(" Evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    main(args.config)
