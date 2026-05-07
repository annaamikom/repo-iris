import mlflow
import os

def load_model():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    token = os.environ.get("MLFLOW_TRACKING_TOKEN")
    model_name = os.environ.get("MODEL_NAME")
    model_stage = os.environ.get("MODEL_STAGE")

    # Set MLflow tracking
    mlflow.set_tracking_uri(tracking_uri)

    # Auth ke DagsHub
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_TOKEN"] = token

    model_uri = f"models:/{model_name}/{model_stage}"

    model = mlflow.pyfunc.load_model(model_uri)

    return model