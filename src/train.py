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
df = pd.read_csv('../data/iris.csv')
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