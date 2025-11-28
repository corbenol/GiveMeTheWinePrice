import requests
import time
import os

# Paramètres de connexion
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "7860")
HEALTH_URL = f"http://{API_HOST}:{API_PORT}/health"
MAX_TRIES = 15
WAIT_SECONDS = 2

print(f"Waiting for API to be ready at {HEALTH_URL}...")

for i in range(MAX_TRIES):
    try:
        response = requests.get(HEALTH_URL, timeout=1)
        if response.status_code == 200:
            print(f"API is ready after {i * WAIT_SECONDS} seconds.")
            break
    except requests.exceptions.ConnectionError:
        print(f"API not ready yet. Retrying in {WAIT_SECONDS} seconds... ({i + 1}/{MAX_TRIES})")
    
    time.sleep(WAIT_SECONDS)
else:
    print("Error: API did not become ready in time.")
    exit(1)

# Laisse l'API en cours d'exécution pour que les tests puissent la contacter