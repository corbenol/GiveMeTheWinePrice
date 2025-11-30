import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder 
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow 
import mlflow.sklearn 
from mlflow.models.signature import infer_signature
import logging
import time
import warnings
import boto3

# fonctions 
def extract_year(text):
    """
    Extrait le millésime (année à 4 chiffres) du titre.
    """
    if pd.isna(text):
        return np.nan
    
    annee_courante = datetime.now().year
    regex_annee = r'(\d{4})'

    # Trouve toutes les occurrences de 4 chiffres
    years = re.findall(regex_annee, str(text))
    
    # Convertit en int et filtre les années plausibles (> 1920 et < année en cours)
    # Note: On conserve le 1920 comme borne basse sécuritaire
    years = [int(y) for y in years if (int(y) > 1920 and int(y) < annee_courante)]
    
    # Retourne la plus grande année trouvée, ou np.nan si aucune
    return max(years) if years else np.nan

def group_rare_categories(df, column, threshold=0.01):
    """
    Regroupe les catégories rares (sous le seuil de fréquence) en 'Other'.
    L'identification des catégories rares se fait sur l'ensemble du DataFrame.
    """
    value_counts = df[column].value_counts(normalize=True)
    rare_categories = value_counts[value_counts < threshold].index
    
    # Créer une copie pour éviter SettingWithCopyWarning
    df_temp = df.copy() 
    df_temp[column] = np.where(df_temp[column].isin(rare_categories), 'Other', df_temp[column])
    return df_temp[column]

def initial_data_cleaning(df):
    """
    Effectue les opérations de nettoyage non intégrables au ColumnTransformer.
    """
    # Suppression des lignes sans prix (cible) ou prix = 0 et sans pays
    df = df.dropna(subset=['price','country'])
    df = df[df['price'] > 0] 

    # Créer la nouvelle colonne avec le log du prix (la cible y)
    df['log_price'] = np.log(df['price']) 
    
    # Extraction du millésime
    df['millesime'] = df['title'].apply(extract_year)

    # Suppression des colonnes spécifiées (inutiles ou trop cardinales)
    COLUMNS_TO_DROP = ['id', 'designation', 'taster_name', 'taster_twitter_handle', 'winery','points','region_1','region_2','title','variety']
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    
    
    # Suppression des incohérences prix/année
    # Cette étape est importante pour la qualité des données et doit être faite avant le split.
    df = df[~((df['millesime'] < 1950) & (df['price'] < 100))]
    
    # Suppression des lignes NA du millésime (car le millésime est essentiel)
    df = df.dropna(subset=['millesime'])
    df['millesime'] = df['millesime'].astype(int).astype(str) # Convertir en string pour traitement catégoriel
    
    # Nettoyage et regroupement des pays (appliqué aux données brutes)
    # On définit ici le regroupement qui sera utilisé plus tard dans le TargetEncoder
    # Le seuil de 1% est un exemple basé sur l'EDA .
    df['country'] = df['country'].str.lower()
    df['province'] = df['province'].str.lower()
    df['country'] = group_rare_categories(df, 'country', threshold=0.01)
    
    return df

def create_preprocessor(df_train):
    """
    Crée le ColumnTransformer pour le Feature Engineering, entraîné sur les données d'entraînement.
    """
    # Définition des groupes de features
    TEXT_FEATURE = 'description'
    TARGET_ENCODING_FEATURES = ['country', 'millesime', 'province']
    
    # 1. Transformation du texte (TF-IDF)
    tfidf_transformer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=500, 
        min_df=5,
        max_df=0.8
    )

    # 2. Transformation catégorielle (Target Encoding)
    # Le TargetEncoder apprend les moyennes de log_price pour 'country', 'millesime', etc.
    # !!! L'intégration de  'province' ajoute de la granularité.
    target_encoder = TargetEncoder(cols=TARGET_ENCODING_FEATURES, smoothing=1.0) 

    # Création du ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', tfidf_transformer, TEXT_FEATURE),          
            ('target_enc', target_encoder, TARGET_ENCODING_FEATURES)
            
        ],
        remainder='drop' 
    )
    return preprocessor

def run_mlops_pipeline_with_mlflow(df_raw,dataset_s3_path):
    
    #TRACK_URI='http://ec2-16-16-98-201.eu-north-1.compute.amazonaws.com:5000'
    TRACK_URI=os.getenv("BACKEND_STORE_URI")
    EXPERIMENT_NAME="Wine_Price_Regression"
    log.info(f"Mlflow tracking, URI : {TRACK_URI}, Experiment : {EXPERIMENT_NAME}")
    # Nom de l'expérience MLflow
    mlflow.set_tracking_uri(TRACK_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_models=True) 

    # Démarre une nouvelle exécution MLflow
    with mlflow.start_run(experiment_id = experiment.experiment_id) as run:
        
                #  PRÉPARATION DES DONNÉES ---
        df_cleaned = initial_data_cleaning(df_raw.copy())
        X = df_cleaned.drop(columns=['price', 'log_price'])
        y = df_cleaned['log_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #  DÉFINITION DU PIPELINE ---
        preprocessor = create_preprocessor(X_train)

        # Définir les hyperparamètres du modèle
        params = {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 5, # Ajout d'un hyperparamètre pour le suivi
            "objective": 'reg:squarederror'
        }
        regressor = XGBRegressor(**params, n_jobs=-1, random_state=42)
        
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])

        # tag le dataset
        mlflow.set_tag("dataset_s3_path", dataset_s3_path)
        log.info(f"Dataset utilisé : {dataset_s3_path}")
        
        # Log des hyperparamètres
        mlflow.log_params(params)
        mlflow.log_param("tfidf_max_features", 500)
        mlflow.log_param("target_encoder_smoothing", 1.0)
        mlflow.log_param("data_split", f"Train={len(X_train)}, Test={len(X_test)}")

        #  ENTRAÎNEMENT ---
        log.info("--- Entraînement du Pipeline Complet... ---")
        full_pipeline.fit(X_train, y_train)

        # : ÉVALUATION ET SUIVI DES MÉTRIQUES ---
        predictions = full_pipeline.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Log des métriques
        mlflow.log_metric("test_rmse_log_scale", rmse)
        mlflow.log_metric("test_r2_score", r2)
        log.info(f"RMSE sur log(price): {rmse:.4f}")
        log.info(f"R² Score: {r2:.4f}")
        
        #  ENREGISTREMENT DE L'ARTEFACT ---
        
        # Enregistrer le pipeline complet comme modèle MLflow (artefact)
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="wine_price_pipeline", 
            registered_model_name="WinePriceRegressorPipeline", 
            signature=infer_signature(X_train, predictions)

        )
        log.info("\n Pipeline enregistré dans MLflow Model Registry.")
        log.info(f"ID d'exécution MLflow : {run.info.run_id}")
        # ---- Définition de l'alias ready_to_test ----
        client = MlflowClient()

        latest_versions = client.get_latest_versions(
            name="WinePriceRegressorPipeline",
            stages=["None"]
        )

        model_version = latest_versions[0].version

        client.set_registered_model_alias(
            name="WinePriceRegressorPipeline",
            alias="ready_to_test",
            version=model_version
        )

        log.info(f"Alias MLflow set : ready_to_test → version {model_version}")
        
    return full_pipeline


# ouverture et chargement du dataset pour train

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    try:
        # ouverture du fichier sur S3
        s3_base = os.getenv("ARTIFACT_DATA")  # ex: "s3://mon_bucket/mon_répertoire/"
        filename = "winemag-train.csv"
        s3_base = s3_base.rstrip('/')  
        parts = s3_base.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        key = f"{prefix}/{filename}" if prefix else filename
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        df_raw = pd.read_csv(obj["Body"])

        #df_raw = pd.read_csv("../data/winemag-data-130k-v2.csv")
    except FileNotFoundError:
        log.error("Erreur : Fichier CSV non trouvé. ")
        exit(3)
    start_time = time.time()
    dataset_s3_path = f"s3://{bucket}/{key}"
    full_pipeline = run_mlops_pipeline_with_mlflow(df_raw,dataset_s3_path)
    #best_pipeline=run_tuning_pipeline_with_mlflow(df_raw)
    log.info(f"---Total training time: {time.time()-start_time}")





