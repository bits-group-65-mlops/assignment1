#!/bin/bash
# This script deploys the Iris Classifier API locally using Docker

echo "Building Docker image..."
docker build -t assignment1:latest .

echo "Starting container..."
docker run -d --name iris-api -p 5001:5001 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 assignment1:latest

echo "Container started. API available at http://localhost:5001"
echo "Test with:"
echo "curl -X POST http://localhost:5001/predict -H 'Content-Type: application/json' -d '{\"data\": [{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}]}'"
echo ""
echo "View metrics at: http://localhost:5001/metrics"
