import requests
import json

# Define the API endpoint
url = "http://localhost:5001/predict"

# Test valid data
valid_data = {
    "data": [
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        {
            "sepal_length": 6.2,
            "sepal_width": 3.4,
            "petal_length": 5.4,
            "petal_width": 2.3
        }
    ]
}

# Test invalid data (negative value)
invalid_data = {
    "data": [
        {
            "sepal_length": -5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    ]
}

# Test missing field
missing_field_data = {
    "data": [
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_width": 0.2
        }
    ]
}

# Send valid request
print("Testing with valid data:")
try:
    response = requests.post(url, json=valid_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting with invalid data (negative value):")
try:
    response = requests.post(url, json=invalid_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting with missing field:")
try:
    response = requests.post(url, json=missing_field_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Get metrics
print("\nGetting metrics:")
try:
    response = requests.get("http://localhost:5001/metrics")
    print(f"Status Code: {response.status_code}")
    print(f"Metrics: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
