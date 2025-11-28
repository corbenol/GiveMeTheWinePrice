import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def test_model_can_be_loaded(mlflow_config, loaded_model):
    """
    Vérifie que le modèle MLflow peut être chargé correctement.
    """
    assert loaded_model is not None, "Le modèle MLflow n'a pas pu être chargé."