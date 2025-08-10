from flask import Flask, request, jsonify
import mlflow
import pandas as pd
import logging
import os
import joblib
from schemas import PredictionRequest

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

app = Flask(__name__)

# Try to load model from MLflow Registry, fallback to local file if not available
model_name = "IrisClassifier"
model_stage = "Production"

try:
    # Attempt to load from MLflow registry
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")
    logging.info(f"Loaded model from MLflow registry: {model_name}/{model_stage}")
except Exception as e:
    logging.warning(f"Could not load model from MLflow registry: {e}")
    
    # Fallback to local model file if it exists
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'iris_classifier.pkl'))
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logging.info(f"Loaded model from local file: {model_path}")
    else:
        # For testing, create a simple model that always returns class 0
        logging.warning("Using dummy model for testing (always predicts class 0)")
        class DummyModel:
            def predict(self, data):
                return [0] * len(data)
        model = DummyModel()

# Simple in-memory metrics store
metrics = {
    "total_requests": 0,
    "predictions_by_class": {0: 0, 1: 0, 2: 0},
    "validation_errors": 0
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        json_data = request.get_json()
        logging.info(f"Received request: {json_data}")
        
        if json_data is None:
            # Handle case where the request doesn't have valid JSON
            logging.error("No JSON data received or Content-Type not set to application/json")
            return jsonify({'error': 'No JSON data received or Content-Type not set to application/json'}), 400
        
        # Validate with Pydantic schema
        try:
            prediction_request = PredictionRequest(**json_data)
            # Convert validated data to dict for DataFrame
            json_data = prediction_request.model_dump()
            logging.info(f"Validated data: {json_data}")
        except Exception as ve:
            metrics["validation_errors"] += 1
            error_message = f"Validation error: {str(ve)}"
            logging.error(error_message)
            return jsonify({'error': error_message}), 400

        # Convert to DataFrame
        df = pd.DataFrame(json_data['data'])
        logging.info(f"DataFrame for prediction: {df.columns.tolist()}")

        # Get prediction
        try:
            prediction = model.predict(df)
            metrics["total_requests"] += 1
            for p in prediction:
                metrics["predictions_by_class"][p] += 1
                
            # Log the prediction and return it
            logging.info(f"Prediction: {prediction.tolist()}")
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            logging.error(error_message)
            return jsonify({'error': error_message}), 400

    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        logging.error(error_message)
        return jsonify({'error': error_message}), 400

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)