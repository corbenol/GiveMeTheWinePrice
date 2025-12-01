import requests
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def test_api_health(health_url):
    """Check that the API health endpoint responds correctly."""
    response = requests.get(health_url, timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"].lower() in ["ok", "healthy"]