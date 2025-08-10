# MLOps Pipeline: Iris Classifier

This project demonstrates a complete MLOps pipeline for training, tracking, versioning, deploying, and monitoring a machine learning model for Iris flower classification.

## 🏛️ Architecture

- **Code Versioning**: Git + GitHub
- **Data Versioning**: DVC
- **Model Tracking & Registry**: MLflow
- **API Framework**: Flask REST API
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Monitoring**: Custom metrics endpoint

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker
- Git
- MLflow server

### 1. Clone & Setup

```bash
# Clone repository
git clone https://github.com/bits-group-65-mlops/assignment1.git
cd assignment1

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Run preprocessing to generate the dataset
python src/preprocess.py
```

### 3. Model Training & Tracking

```bash
# Start MLflow tracking server (in a separate terminal)
mlflow server --host 127.0.0.1 --port 5000

# Train models and track with MLflow
python src/train.py

# Go to http://127.0.0.1:5000 to view experiments
```

### 4. API Deployment (Local)

```bash
# Run the Flask API directly
python src/app.py

# Test with curl
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}'
```

### 5. Docker Deployment

```bash
# Build Docker image
docker build -t assignment1:latest .

# Run container
docker run -p 5001:5001 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 assignment1:latest

# Access metrics
curl http://localhost:5001/metrics
```

## 📊 Project Structure

```
├── .github/workflows/   # GitHub Actions CI/CD pipeline
├── data/                # Dataset files
│   ├── iris.csv         # Iris dataset
│   └── iris.csv.dvc     # DVC tracking file
├── src/                 # Source code
│   ├── app.py           # Flask API
│   ├── preprocess.py    # Data preprocessing
│   └── train.py         # Model training script
├── tests/               # Unit tests
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 🧪 Testing

```bash
# Run tests
pytest tests/
```

## 📝 CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Lints and tests the code
2. Builds the Docker image
3. Pushes to Docker Hub

## 📝 License

MIT
