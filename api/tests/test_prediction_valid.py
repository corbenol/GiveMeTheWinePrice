import requests


def test_api_prediction_valid_input(predict_url):
    """Send a valid input payload and verify prediction result."""
    sample_input = {
        "fixed_acidity": 7.2,
        "volatile_acidity": 0.35,
        "citric_acid": 0.45,
        "residual_sugar": 5.1,
        "chlorides": 0.05,
        "free_sulfur_dioxide": 22,
        "total_sulfur_dioxide": 130,
        "density": 0.995,
        "pH": 3.2,
        "sulphates": 0.65,
        "alcohol": 10.5,
    }

    response = requests.post(predict_url, json=sample_input, timeout=5)

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))