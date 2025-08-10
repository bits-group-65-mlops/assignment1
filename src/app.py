from flask import Flask, request, jsonify
import mlflow
import pandas as pd
import logging
from schemas import PredictionRequest

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

app = Flask(__name__)

# Load the model from the MLflow Model Registry
model_name = "IrisClassifier"
model_stage = "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

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
        
        # Validate with Pydantic schema
        try:
            prediction_request = PredictionRequest(**json_data)
            # Convert validated data to dict for DataFrame
            json_data = prediction_request.dict()
        except Exception as ve:
            metrics["validation_errors"] += 1
            error_message = f"Validation error: {str(ve)}"
            logging.error(error_message)
            return jsonify({'error': error_message}), 400

        # Convert to DataFrame
        df = pd.DataFrame(json_data['data'])

        # Get prediction
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

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)