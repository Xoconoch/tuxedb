# Use a lightweight Python base image
FROM python:3.12-slim

# Install dependencies
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the application files
COPY requirements.txt .
COPY analisis.py .
COPY app.py .
COPY entrypoint.sh /entrypoint.sh

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Expose the application port
EXPOSE 5000

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "app.py"]
