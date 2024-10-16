# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Set the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
