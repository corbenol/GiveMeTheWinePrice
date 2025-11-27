import pandas as pd
import numpy as np

def test_model_can_predict(loaded_model):
    """
    Vérifie que le modèle peut prédire sur un exemple simple.
    """
    sample = pd.DataFrame([{
        "country": "france",
        "description": "fruity and elegant wine",
        "province": "alsace",
        "millesime": "2018"
    }])

    prediction = loaded_model.predict(sample)

    assert len(prediction) == 1, "Le modèle doit renvoyer une seule prédiction."
    assert isinstance(prediction[0], (int, float,np.float32, np.float64)), "La prédiction doit être un nombre."