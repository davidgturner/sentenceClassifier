import mlflow

# Get the list of runs for the experiment
runs = mlflow.search_runs(experiment_ids="dataset_versioning")

# Retrieve the artifacts (datasets) for a specific run
run_id = runs.iloc[0]["run_id"]
artifacts_uri = mlflow.get_tracking_uri() + "/experiments/dataset_versioning/runs/" + run_id + "/datasets"
artifacts = mlflow.list_artifacts(artifacts_uri)

# Access a specific version of the dataset
version_path = artifacts[0].path
full_dataset_path = mlflow.get_artifact_uri() + version_path