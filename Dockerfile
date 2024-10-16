FROM python:3.12

# Install required packages
RUN apt-get update && apt-get install -y python3-distutils

# Set the working directory
WORKDIR /mount/src/meducation1

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
