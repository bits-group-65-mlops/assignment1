import pandas as pd
from sklearn.datasets import load_iris
import os

# Load the dataset
iris = load_iris(as_frame=True)
df = iris.frame
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

# For simplicity, we'll just save it.
# In a real project, you'd split and process it here.
# Create data directory if it doesn't exist
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(data_dir, exist_ok=True)

# Save the dataset to the data directory
data_path = os.path.join(data_dir, 'iris.csv')
df.to_csv(data_path, index=False)
print(f"Dataset saved to {data_path}")