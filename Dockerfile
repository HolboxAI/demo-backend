# Use Python 3.12 slim as base image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for pyaudio, psycopg2, and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    portaudio19-dev \
    python3-dev \
    libgl1 \
    unzip \
    mariadb-server \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . .

# Create a virtual environment in /app/venv
RUN python3 -m venv /app/venv

# Upgrade pip inside the virtual environment
RUN /app/venv/bin/pip install --upgrade pip

# Install the dependencies from requirements.txt inside the virtual environment
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Expose the port that uvicorn will use (default 8000)
EXPOSE 8000

# Command to run the application with uvicorn
CMD ["/app/venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]