from typing import  Union
import pandas as pd
import numpy as np
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from starlette.responses import RedirectResponse
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from fastapi.responses import JSONResponse


# documentation
description = """
Bienvenue sur notre API pour estimer le prix du vin en fonction de  
* son pays  
* sa description oenologique  
* sa région  
* son millesime (4 digits)  

##  Endpoints
* `/`: Redirige vers la page `/docs`
* `/health`:  état du service
## Machine Learning
* `/predict`:**POST**   (country : str | description : str | province : str | millesime : str (4 chiffres))

"""

tags_metadata = [
    {"name": "Accueil", "description": "Page d'accueil" },
    {"name": "Check", "description": "vérification de la disponibilité du service"},
    {"name": "Machine Learning", "description": "Accès au service de prédiction" }
]

# chargement des variables stockées dans l'environnement HF space
# les codes acces S3 sont captés directement de l'environnement.
NEON_URI=os.getenv("NEON_URI")
MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")

print(f'MLFLOW : {MLFLOW_TRACKING_URI}')
print(f'neon : {NEON_URI}')

# --- Configuration MLOps ---
MODEL_NAME = "WinePriceRegressorPipeline"
MODEL_STAGE = "production" 

# fonctions
def get_db_conn():
    return psycopg2.connect(NEON_URI, cursor_factory=RealDictCursor)

class WineFeatures(BaseModel):
    country: str 
    description: str 
    province: str 
    millesime: str 
    @field_validator('country', 'description', 'province', mode='before')
    @classmethod
    def to_lower(cls, value: str):
        if isinstance(value, str):
            return value.lower()
        return value 
    
# --- Chargement du Modèle au Démarrage ---
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Charge le modèle le plus récent aliasé 'Production'
    logged_model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
    print(logged_model_uri)
    model = mlflow.pyfunc.load_model(logged_model_uri)
    print("Modèle chargé avec succès depuis MLflow.")
except Exception as e:
    print(f"ÉCHEC DU CHARGEMENT DU MODÈLE MLFLOW : {e}")
    model = None 
    exit(3)

# --- Création automatique de la table dans Neon ---
def init_db():
    create_table_query = """
    CREATE TABLE IF NOT EXISTS wine_predictions (
        id SERIAL PRIMARY KEY,
        country TEXT,
        description TEXT,
        province TEXT,
        millesime TEXT,
        predicted_price NUMERIC,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute(create_table_query)
        conn.commit()
        cur.close()
        conn.close()
        print("Table wine_predictions OK")
    except Exception as e:
        print(f"Erreur création table : {e}")

init_db()


app = FastAPI(title="Quel prix pour ce vin ?",
    description=description,
    version="0.1",
    openapi_tags=tags_metadata
)

@app.get("/", tags=["Accueil"])
async def index():
    """
    Redirige la page d'accueil vers la documentation interactive de l'API (/docs).
    """
    # Renvoie une réponse de redirection temporaire vers l'URL /docs
    return RedirectResponse(url="/docs")
@app.get("/health", tags=["Check"])
def health_check():
    """ Vérifie que le service est en ligne et que le modèle est chargé. """
    status = "OK" if model is not None else "Model Loading Failed"
    return {"status": status, "model_name": MODEL_NAME}

@app.post("/predict",tags=["Machine Learning"])
async def predict_price(features: WineFeatures):
    """
    Accès au service de prédiction du prix du vin
    """
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Modèle non chargé, vérifiez la connexion MLflow."}
        )
    # Conversion des données entrantes en DataFrame (requis par le pipeline)
    input_data = pd.DataFrame([features.model_dump()])
    # Prédiction sur l'échelle Log
    log_prediction = model.predict(input_data)[0]
    #Retour à l'échelle réelle, np.log lors du training
    price_prediction = np.exp(log_prediction) 
    price_numeric = round(float(price_prediction), 2)
    # --- Enregistrement dans Neon ---
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        insert_query = """
            INSERT INTO wine_predictions 
            (country, description, province, millesime, predicted_price)
            VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(
            insert_query,
            (
                features.country,
                features.description,
                features.province,
                features.millesime,
                price_numeric
            )
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Erreur insertion Neon : {e}")

    return {"prediction": price_numeric}