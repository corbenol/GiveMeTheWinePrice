import os
import pytest
import mlflow
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



@pytest.fixture(scope="session")
def mlflow_config():
    """
    Récupère la configuration MLflow depuis les variables d'environnement.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = os.getenv("MODEL_NAME", "WinePriceRegressorPipeline")
    model_stage = os.getenv("MODEL_STAGE", "Staging")  # staging == ready_to_test

    assert tracking_uri is not None, "MLFLOW_TRACKING_URI n'est pas défini"

    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}@{model_stage}"

    return {
        "tracking_uri": tracking_uri,
        "model_name": model_name,
        "model_stage": model_stage,
        "model_uri": model_uri
    }


@pytest.fixture(scope="session")
def loaded_model(mlflow_config):
    """
    Charge une fois le modèle MLflow pour tous les tests.
    """
    model = mlflow.pyfunc.load_model(mlflow_config["model_uri"])
    return model