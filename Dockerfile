# Use the official Python base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose JupyterLab's default port
EXPOSE 8888