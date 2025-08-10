import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris Classification")

# Load data
import os
# Get the absolute path to the data directory
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'iris.csv'))
df = pd.read_csv(data_path)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Logistic Regression ---
with mlflow.start_run(run_name='Logistic Regression'):
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Logistic Regression Accuracy: {accuracy:.2f}")

    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(lr, "model")

# --- Train Random Forest ---
with mlflow.start_run(run_name='Random Forest'):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=1, max_features='sqrt')
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Random Forest Accuracy: {accuracy:.2f}")

    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("min_samples_leaf", 1)
    mlflow.log_param("max_features", "sqrt")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(rf, "model")

# --- Register the best model in Model Registry ---
# Get the best run by accuracy
best_run = mlflow.search_runs(
    experiment_ids=mlflow.get_experiment_by_name("Iris Classification").experiment_id,
    order_by=["metrics.accuracy DESC"]
).iloc[0]

# Register the model from the best run
model_uri = f"runs:/{best_run.run_id}/model"
try:
    # Check if model already exists
    client = mlflow.tracking.MlflowClient()
    try:
        existing_model = client.get_registered_model("IrisClassifier")
        print(f"Model 'IrisClassifier' already exists. Registering new version.")
    except:
        print(f"Creating new model 'IrisClassifier'")
    
    mv = mlflow.register_model(model_uri, "IrisClassifier")
    
    # Transition the model to Production
    client.transition_model_version_stage(
        name="IrisClassifier",
        version=mv.version,
        stage="Production"
    )
    
    print(f"Model registered as: {mv.name} version {mv.version}")
    print(f"Model transitioned to Production stage")
    
    # Save the best model locally for CI/CD environments without MLflow
    import joblib
    best_model = mlflow.sklearn.load_model(model_uri)
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, 'iris_classifier.pkl'))
    print(f"Model saved to {os.path.join(models_dir, 'iris_classifier.pkl')}")
    
except Exception as e:
    print(f"Error registering model: {e}")
    print("You can register the model manually via the MLflow UI at http://127.0.0.1:5000")
    
    # Even if MLflow registration fails, save the best model based on accuracy
    if 'rf' in locals() and 'lr' in locals():
        import joblib
        # Compare accuracies and save the best model
        rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
        lr_accuracy = accuracy_score(y_test, lr.predict(X_test))
        best_model = rf if rf_accuracy >= lr_accuracy else lr
        
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'iris_classifier.pkl')
        joblib.dump(best_model, model_path)
        print(f"Saved best model to {model_path} for CI/CD environments")