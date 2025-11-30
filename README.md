# 🍷 Give Me The Wine Price - Application de Prédiction du Prix du Vin

Ce dépôt (`corbenol/GiveMeTheWinePrice`) contient l'intégralité de la chaîne MLOps pour l'entraînement, la validation, le monitoring et le déploiement d'un modèle de régression du prix du vin.

L'application d'inférence est une API **FastAPI** qui utilise **MLflow** pour charger le modèle de production et **Neon (PostgreSQL)** pour la journalisation des requêtes (Inference Logging).

## 🚀 Architecture MLOps et Composants

L'architecture est construite sur une séparation claire des environnements :

| **Composant** | **Rôle** | **Technologie** | **Répertoire** | **Environnement** |
| :--- | :--- | :--- | :--- | :--- |
| **Tracking Server** | Serveur central pour le suivi des expériences et la gestion des modèles. | MLflow / Docker | `tracker_aws/` | **EC2 AWS** |
| **Entraînement** | Exécution de l'entraînement, du logging et de l'enregistrement du modèle. | Docker / Python | `model/` | **EC2 via SSH/GitHub Actions** |
| **Monitoring** | Surveillance de la dérive (drift) du modèle en comparant la performance en production à la performance de référence. | Evidently AI / Python | `drift/` | **GitHub Actions** |
| **API Web (Inférence)** | Prédiction et journalisation des données dans Neon DB. | FastAPI / MLflow | `api/` | **Hugging Face Spaces** |

## 🛠️ Infrastructure et Fichiers Clés

| **Fichier/Répertoire** | **Description** | **Utilisé par Workflow** |
| :--- | :--- | :--- |
| `model/train.py` | Script principal d'entraînement et d'enregistrement du modèle dans MLflow. | `mlflow-train.yml` |
| `model/Dockerfile.train` | Image Docker pour exécuter l'entraînement sur EC2. | `mlflow-train.yml`, `mlflow-test.yml` |
| `api/app.py` | Code de l'API FastAPI, incluant la connexion à Neon DB et le chargement du modèle de production. | `api-deploy.yml` |
| `api/Dockerfile` | Image Docker pour l'API web (utilisée par Hugging Face Spaces). | `api-deploy.yml` |
| `drift/drift_monitor.py` | Script de monitoring qui compare la RMSE de production aux métriques de référence MLflow. | `drift_monitoring.yaml` |
| `.github/workflows/` | Contient les quatre workflows CI/CD. | Tous |

## ⚙️ Workflows MLOps avec GitHub Actions

Les pipelines CI/CD sont gérés par les tags Git pour contrôler précisément les étapes d'entraînement, de test, de déploiement et de monitoring.

| **Fichier Workflow** | **Déclencheur (Tag ou Événement)** | **Étapes Clés** |
| :--- | :--- | :--- |
| **`mlflow-train.yml`** | Tag `train-*` (ex: `train-v1.2`) | Configure SSH, synchronise le code avec EC2, construit l'image Docker (`model/Dockerfile.train`) sur EC2, et exécute `model/train.py` via SSH. |
| **`mlflow-test.yml`** | Succès de `mlflow-train.yml` | Construit l'image de test, exécute `pytest`, et si les tests réussissent, **proscrit l'alias MLflow de la version testée à `production`**. |
| **`api-deploy.yml`** | Tag `deploy-*` (ex: `deploy-20251130`) | Exécute des tests d'intégration (via une image Docker de test), puis utilise la CLI `huggingface_hub` pour mettre à jour le Space `corbenol/wine-price-predictor`. |
| **`drift_monitoring.yaml`** | Tag `drift-*` (ou calendrier) | Construit l'image Docker de dérive, exécute `drift/drift_monitor.py` pour évaluer la performance du modèle en production et échoue si la dérive dépasse le seuil défini. |

## ☁️ Déploiement et Configuration

### 1. Configuration des Secrets GitHub (Environnement)

Tous les workflows nécessitent la configuration des secrets suivants dans les paramètres de votre dépôt GitHub (`Settings > Secrets > Actions`):

| **Secret** | **Utilisé par** | **Rôle** |
| :--- | :--- | :--- |
| `EC2_SSH_PRIVATE_KEY` | `mlflow-train.yml` | Clé SSH privée pour la connexion et l'exécution de Docker sur le serveur EC2. |
| `EC2_HOST` | `mlflow-train.yml` | Adresse IP ou DNS du serveur d'entraînement EC2. |
| `BACKEND_STORE_URI` | Tous les workflows | URI du serveur MLflow (ex: `http://ec2-xx-xx-xx-xx:5000/`). |
| `AWS_ACCESS_KEY_ID` | Tous les workflows | Clé d'accès S3 (artefacts MLflow et données de drift). |
| `AWS_SECRET_ACCESS_KEY` | Tous les workflows | Clé secrète d'accès S3. |
| `NEON_TEST` | `api-deploy.yml` | URI de la base de données Neon pour les tests d'intégration de l'API. |
| `HF_TOKEN` | `api-deploy.yml` | Jeton d'accès Hugging Face pour la mise à jour du Space. |
| `ARTIFACT_DATA` | `mlflow-train.yml`, `drift_monitoring.yaml` | Chemin S3 vers les données brutes (ex: `s3://mon-bucket-wine-data/`). |

### 2. Démarrage du Serveur de Suivi MLflow (EC2)

Le serveur de suivi est déployé dans un conteneur Docker sur EC2. Pour le démarrer :

1. Connectez-vous à votre instance EC2.

2. Dans le répertoire `tracker_aws/`, exécutez le script `build.sh` pour créer l'image Docker.

3. Exécutez `run_docker.sh` pour lancer le conteneur, en configurant les variables d'environnement nécessaires (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc.).

### 3. Déploiement de l'API (Hugging Face)

Le déploiement est automatique après un tag `deploy-*` et l'exécution réussie des tests d'intégration dans `api-deploy.yml`. Ce workflow copie les fichiers essentiels (`api/app.py`, `api/requirements-api.txt`, `api/Dockerfile`) et les pousse vers le Space Hugging Face cible (`corbenol/wine-price-predictor`).

## 🔎 Comment Utiliser et Tester l'API (Inférence)

L'API est accessible sur votre Space Hugging Face (ex: `https://corbenol-wine-price-predictor.hf.space`).

### Endpoint de Santé (`/health`)

Vérifie l'état du service et confirme la version du modèle chargé à partir de l'alias `production` de MLflow.

### Endpoint de Prédiction (`/predict`)

**Méthode:** `POST`

**Schéma de la Requête (JSON):**

```json
{
    "country": "france",
    "description": "Un vin rouge avec des notes de cerise et de tanins légers.",
    "province": "bordeaux",
    "millesime": "2018" 
}