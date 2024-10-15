# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Run the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
