FROM python:3.11

# หรือใช้ python:3.12 ถ้าคุณต้องการ

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install distutils and then install the dependencies
RUN apt-get update && apt-get install -y python3-distutils
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Command to run your application
CMD ["gunicorn", "your_app:app", "-b", "0.0.0.0:5000"]
