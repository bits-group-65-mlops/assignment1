# Stage 1: Install dependencies
FROM python:3.9-slim as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Create the final image
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY src/ /app/

# Expose the port the app runs on
EXPOSE 5001

# Command to run the application
# Note: Ensure the MLFLOW_TRACKING_URI is set when running the container
CMD ["python", "app.py"]
