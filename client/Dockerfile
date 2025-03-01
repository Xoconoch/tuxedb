# Use a lightweight Python base image
FROM python:3.12-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    libc6-dev


# Set the working directory
WORKDIR /app

# Copy the application files
COPY requirements.txt .
COPY client.py .
COPY entrypoint.sh /entrypoint.sh

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "client.py"]
