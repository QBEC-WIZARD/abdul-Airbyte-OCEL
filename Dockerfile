# Use an official Python image as base
FROM python:3.10-slim

# Set environment variables to reduce pip issues
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

# Set work directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install dependencies with increased timeout and retries
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --retries=10 -r requirements.txt

# Default command to run the Airbyte connector
CMD ["python", "main.py"]
