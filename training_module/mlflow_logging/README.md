# DagsHub MLflow Logging

This folder logs completed training result folders to DagsHub MLflow.

## Configure Credentials

`mlflow_config.yaml` contains private DagsHub credentials, so it is ignored by Git.
Use the committed example file as the template:

```bash
cp training_module/mlflow_logging/mlflow_config_example.yaml \
  training_module/mlflow_logging/mlflow_config.yaml
```

Then edit `mlflow_config.yaml` and replace the placeholder values:

```yaml
dagshub:
  repo_owner: "your-dagshub-username"
  repo_name: "your-dagshub-repo"
  username: "your-dagshub-username"
  token: "your-dagshub-token"

mlflow:
  experiment_name: "face-recognition-training"
  run_name: ""
  artifact_path: "training_results"
  tracking_uri: ""

model:
  checkpoint_name: "best.pth"
  artifact_path: "model"
  registered_model_name: "face-recognition-arcface"
  normalize_output: true
```

Keep `mlflow_config_example.yaml` in Git, but do not commit `mlflow_config.yaml`.

## Run Logging

Install dependencies:

```bash
pip install mlflow dagshub pyyaml torch
```

Log a result folder:

```bash
python training_module/mlflow_logging/logging.py \
  training_module/mlflow_logging/face_recognition_results
```

If no folder is passed, the script asks for one. A valid folder should follow the same shape as `face_recognition_results` and include:

```text
training_history.json
training_metadata.json
plots or other result artifacts
```

The script logs config and dataset metadata as params, logs only the `epoch: "test"` row from `training_history.json` as test metrics, uploads non-model result files as artifacts, and registers `best.pth` as an MLflow PyTorch model so it appears in the DagsHub Models tab.
