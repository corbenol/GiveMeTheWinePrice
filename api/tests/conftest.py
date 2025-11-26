import os
import pytest

API_URL = os.getenv("API_URL")
HEALTH_ENDPOINT = os.getenv("API_HEALTH_ENDPOINT", "/health")
PREDICT_ENDPOINT = os.getenv("API_PREDICT_ENDPOINT", "/predict")


@pytest.fixture(scope="session")
def base_url():
    """Base API URL loaded from environment."""
    if not API_URL:
        raise RuntimeError("Environment variable API_URL not set.")
    return API_URL.rstrip("/")


@pytest.fixture(scope="session")
def health_url(base_url):
    return f"{base_url}{HEALTH_ENDPOINT}"


@pytest.fixture(scope="session")
def predict_url(base_url):
    return f"{base_url}{PREDICT_ENDPOINT}"