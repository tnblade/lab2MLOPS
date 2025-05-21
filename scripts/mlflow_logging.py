import mlflow

mlflow.set_experiment("Telco Churn Evaluation")

with mlflow.start_run():
    mlflow.log_metric("default_model_acc", acc_default)
    mlflow.log_metric("best_model_acc", acc_best)
    mlflow.log_param("selected_model", best_model_name)
    mlflow.log_artifact("models/deploy_model.pkl")
