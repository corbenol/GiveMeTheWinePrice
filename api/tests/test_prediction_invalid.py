import requests

def test_api_prediction_invalid_input(predict_url):
    """Ensure API returns a clean validation error for malformed payloads."""
    invalid_input = {"foo": "bar"}

    response = requests.post(predict_url, json=invalid_input, timeout=5)

    # 400/422 expected depending on API implementation
    assert response.status_code in (400, 422)

    data = response.json()

    # FastAPI uses "detail", others may return "error"
    assert "error" in data or "detail" in data