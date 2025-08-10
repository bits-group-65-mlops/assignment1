# PowerShell script to deploy the Iris Classifier API locally using Docker

Write-Host "Building Docker image..." -ForegroundColor Green
docker build -t assignment1:latest .

Write-Host "Starting container..." -ForegroundColor Green
docker run -d --name iris-api -p 5001:5001 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 assignment1:latest

Write-Host "Container started. API available at http://localhost:5001" -ForegroundColor Cyan
Write-Host "Test with:" -ForegroundColor Yellow
Write-Host "Invoke-RestMethod -Method Post -Uri http://localhost:5001/predict -ContentType 'application/json' -Body '{\"data\": [{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}]}'" -ForegroundColor Yellow
Write-Host ""
Write-Host "View metrics at: http://localhost:5001/metrics" -ForegroundColor Cyan
