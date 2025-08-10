#!/usr/bin/env python
"""
This script creates a simple model for CI/CD environments without requiring MLflow.
It is used in the GitHub Actions workflow to generate a model for testing.
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os

print('Training model for CI/CD...')
iris = load_iris()
# Using underscore feature names to match the API input
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = pd.DataFrame(iris.data, columns=feature_names)
y = iris.target
model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=1, max_features='sqrt')
model.fit(X, y)
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/iris_classifier.pkl')
print('Model saved for CI/CD tests')
