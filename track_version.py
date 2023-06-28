import mlflow

# Log the dataset as an artifact in MLflow
dataset_path = "path/to/dataset.csv"
with mlflow.start_run():
    mlflow.log_artifact(dataset_path, "datasets")