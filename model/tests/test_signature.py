import mlflow

def test_model_signature_is_valid(mlflow_config):
    """
    Valide que la signature MLflow contient les bons champs d'entrée.
    """
    model_info = mlflow.models.get_model_info(mlflow_config["model_uri"])
    signature = model_info.signature

    assert signature is not None, "Aucune signature MLflow trouvée."

    expected_inputs = {"country", "description", "province", "millesime"}
    actual_inputs = {col.name for col in signature.inputs}

    missing = expected_inputs - actual_inputs
    assert not missing, f"Colonnes manquantes dans la signature : {missing}"