import pytest
from flask import json
import sys
import os

# Add the src directory to the path so we can import the app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the app, mocking the MLflow model loading
import app as flask_app
import mlflow.pyfunc

# Mock the MLflow model
class MockModel:
    def predict(self, data):
        # Just return class 0 for all predictions in tests
        return [0] * len(data)

# Patch MLflow's load_model to return our mock
mlflow.pyfunc.load_model = lambda model_uri: MockModel()

@pytest.fixture
def app():
    flask_app.app.config.update({
        "TESTING": True,
    })
    return flask_app.app

@pytest.fixture
def client(app):
    return app.test_client()

def test_predict_endpoint(client):
    """Test the predict endpoint with valid data."""
    response = client.post(
        '/predict',
        data=json.dumps({
            'data': [
                {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2}
            ]
        }),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "prediction" in data
    assert isinstance(data["prediction"], list)

def test_metrics_endpoint(client):
    """Test the metrics endpoint."""
    response = client.get('/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "total_requests" in data
    assert "predictions_by_class" in data
