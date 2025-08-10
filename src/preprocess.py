import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the dataset
iris = load_iris(as_frame=True)
df = iris.frame
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

# For simplicity, we'll just save it.
# In a real project, you'd split and process it here.
df.to_csv('data/iris.csv', index=False)
print("Dataset saved to data/iris.csv")