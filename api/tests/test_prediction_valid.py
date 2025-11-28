import requests


def test_api_prediction_valid_input(predict_url):
    """Send a valid input payload and verify prediction result."""
    sample_input = {
        "country": "france",
        "description": "fruity and elegant wine",
        "province": "alsace",
        "millesime": "2022"
    }

    response = requests.post(predict_url, json=sample_input, timeout=5)

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))