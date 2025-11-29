import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
from mlflow.tracking import MlflowClient
import mlflow
import boto3
import logging
import warnings

# --- Configuration et Initialisation ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Constantes pour la connexion MLflow ---

TRACK_URI=os.getenv("TRACK_URI")
REGISTERED_MODEL_NAME = "WinePriceRegressorPipeline"
# Alias MLflow pour le modèle en production 
MODEL_ALIAS = "production" 
# Nom du fichier de données de test/surveillance dans S3
DATA_FILENAME = "winemag-test.csv" 
# Nom de la métrique de référence à comparer 
METRIC_NAME = "test_rmse_log_scale" 

# --- Fonctions de Nettoyage de Données (Copie de l'entraînement) ---

def extract_year(text):
    """ Extrait le millésime (année à 4 chiffres) du titre. """
    if pd.isna(text):
        return np.nan
    annee_courante = datetime.now().year
    years = [int(y) for y in re.findall(r'(\d{4})', str(text)) 
             if (int(y) > 1920 and int(y) < annee_courante)]
    return max(years) if years else np.nan

def group_rare_categories(df, column, threshold=0.01):
    """ Regroupe les catégories rares (sous le seuil) en 'Other'. """
    value_counts = df[column].value_counts(normalize=True)
    rare_categories = value_counts[value_counts < threshold].index
    df_temp = df.copy() 
    df_temp[column] = np.where(df_temp[column].isin(rare_categories), 'Other', df_temp[column])
    return df_temp[column]

def initial_data_cleaning(df):
    """
    Effectue les opérations de nettoyage nécessaires avant la prédiction.
    NOTE: Les étapes de TARGET ENCODING et TF-IDF sont gérées par le pipeline chargé.
    """
    # 1. Suppression des lignes sans prix (cible) ou prix = 0 et sans pays
    # NOTE: Pour la surveillance de dérive, nous allons prédire le prix
    # mais nous avons besoin de la colonne 'price' pour calculer la nouvelle RMSE
    df = df.dropna(subset=['price','country'])
    df = df[df['price'] > 0] 

    # 2. Créer la nouvelle colonne avec le log du prix (la cible y)
    df['log_price'] = np.log(df['price']) 
    
    # 3. Extraction du millésime
    df['millesime'] = df['title'].apply(extract_year)

    # 4. Suppression des colonnes spécifiées
    COLUMNS_TO_DROP = ['id', 'designation', 'taster_name', 'taster_twitter_handle', 'winery','points','region_1','region_2','title','variety']
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    
    # 5. Suppression des incohérences prix/année et des NA de millésime
    df = df[~((df['millesime'] < 1950) & (df['price'] < 100))]
    df = df.dropna(subset=['millesime'])
    df['millesime'] = df['millesime'].astype(int).astype(str)
    
    # 6. Nettoyage et regroupement des pays/province (DOIT correspondre à l'entraînement)
    df['country'] = df['country'].str.lower()
    df['province'] = df['province'].str.lower()
    # IMPORTANT: Le TargetEncoder dans le pipeline gérera les nouvelles valeurs vues 
    # dans group_rare_categories si elles n'ont pas été vues à l'entraînement.
    df['country'] = group_rare_categories(df, 'country', threshold=0.01)
    
    return df

# --- Logique MLflow et S3 ---

def get_s3_data(s3_path: str) -> pd.DataFrame:
    """ Télécharge le fichier de test depuis S3. """
    log.info(f"Tentative de chargement des données depuis S3: {s3_path}")
    
    try:
        s3_path = s3_path.rstrip('/') 
        parts = s3_path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        key = f"{prefix}/{DATA_FILENAME}" if prefix else DATA_FILENAME
        
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj["Body"])
        log.info(f"Données chargées avec succès. Taille: {len(df)} lignes.")
        return df
    except Exception as e:
        log.error(f"Erreur lors du chargement des données S3: {e}")
        # Termine le programme si les données ne sont pas accessibles
        raise

def load_mlflow_model_and_metrics(model_name: str, alias: str):
    """ 
    Charge le modèle de production et récupère la métrique de référence.
    Retourne le pipeline chargé et la métrique de référence (RMSE log).
    """
    log.info(f'serveur MLFLOW : {TRACK_URI}')
    mlflow.set_tracking_uri(TRACK_URI)
    client = MlflowClient()

    # 1. Charger le pipeline (basé sur l'alias, e.g., 'production')
    model_uri = f"models:/{model_name}@{alias}"

    log.info(f"Chargement du pipeline de production depuis: {model_uri}")
    full_pipeline = mlflow.sklearn.load_model(model_uri)
    log.info(f'pipeline : {full_pipeline}')
    
    # 2. Récupérer la version du modèle pour trouver la métrique de référence
    try:
        # Tente de récupérer la version par alias
        version_info = client.get_model_version_by_alias(model_name, alias)
        version_info_list = [version_info]
        if not version_info_list:
             # Si l'alias ne renvoie rien, essaie de récupérer la dernière version tout court.
            version_info_list = client.get_latest_versions(model_name, stages=[alias])
            if not version_info_list:
                raise ValueError(f"Aucune version trouvée pour l'alias/stage '{alias}'")

        mv = version_info_list[0]  
        version = mv.version
        run_id = mv.run_id

        # 3. Récupérer la métrique de référence (RMSE log)
        metric_value = client.get_run(run_id).data.metrics.get(METRIC_NAME)
        
        if metric_value is None:
            log.error(f"Métrique de référence '{METRIC_NAME}' non trouvée pour le Run ID {run_id}.")
            raise ValueError("Métrique de référence manquante.")
            
        log.info(f"Modèle version {version} ({alias}) chargé. RMSE de référence: {metric_value:.4f}")
        return full_pipeline, metric_value

    except Exception as e:
        log.error(f"Erreur lors de la récupération des métriques MLflow: {e}")
        raise

# --- Logique de Dérive ---

def check_model_drift(df_raw, full_pipeline, reference_metric: float, drift_threshold: float):
    """
    Exécute la prédiction, calcule la nouvelle métrique et compare à la référence.
    """
    from sklearn.metrics import mean_squared_error

    # 1. Préparation des données
    df_test = initial_data_cleaning(df_raw.copy())
    X_test = df_test.drop(columns=['price', 'log_price'], errors='ignore')
    y_test_log = df_test['log_price']
    
    # 2. Prédiction
    log.info(f"Début de la prédiction sur {len(X_test)} échantillons...")
    predictions_log = full_pipeline.predict(X_test)
    
    # 3. Calcul de la nouvelle métrique
    new_rmse_log = np.sqrt(mean_squared_error(y_test_log, predictions_log))

    log.info("-" * 50)
    log.info(f"Métrique de Référence ({METRIC_NAME}) : {reference_metric:.4f}")
    log.info(f"Nouvelle Métrique Calculée : {new_rmse_log:.4f}")
    log.info(f"Seuil de Dérive (Tolérance) : {drift_threshold * 100}%")

    # 4. Comparaison de la dérive
    # Calcul de la variation en pourcentage par rapport à la référence
    drift_percentage = abs(new_rmse_log - reference_metric) / reference_metric
    
    if drift_percentage > drift_threshold:
        log.warning(f"!!! DÉRIVE DÉTECTÉE !!!")
        log.warning(f"La RMSE a augmenté de {drift_percentage*100:.2f}% (Seuil: {drift_threshold*100:.2f}%)")
        log.warning("Le modèle nécessite probablement un réentraînement ou une inspection des données.")
        # Utiliser exit(1) pour signaler l'échec du pipeline CI/CD
        exit(1) 
    else:
        log.info("PAS DE DÉRIVE DÉTECTÉE. Performance stable.")
        log.info(f"Variation RMSE: {drift_percentage*100:.2f}% (sous le seuil de {drift_threshold*100:.2f}%)")
        # Utiliser exit(0) pour signaler le succès du pipeline CI/CD
        exit(0) 

def main():
    # Variables d'environnement pour S3 et le seuil
    s3_artifact_data = os.getenv("ARTIFACT_DATA") 
    drift_threshold_str = os.getenv("DRIFT_THRESHOLD", "0.10") # 10% par défaut
    
    if not s3_artifact_data:
        log.error("La variable d'environnement ARTIFACT_DATA est manquante.")
        exit(1)

    try:
        drift_threshold = float(drift_threshold_str)
    except ValueError:
        log.error(f"DRIFT_THRESHOLD '{drift_threshold_str}' n'est pas un nombre valide.")
        exit(1)

    try:
        # 1. Chargement du modèle MLflow et de la métrique de référence
        pipeline, ref_metric = load_mlflow_model_and_metrics(REGISTERED_MODEL_NAME, MODEL_ALIAS)
        
        # 2. Chargement des nouvelles données de test S3
        df_raw = get_s3_data(s3_artifact_data)
        
        # 3. Vérification de la dérive
        check_model_drift(df_raw, pipeline, ref_metric, drift_threshold)

    except Exception as e:
        log.error(f"Échec critique du processus de surveillance de la dérive: {e}")
        # Termine avec un code d'erreur pour que le pipeline CI/CD échoue
        exit(1) 


if __name__ == '__main__':
    main()