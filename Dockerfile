# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /usr/src/app

# Copy requirements and source files
COPY requirements.txt ./
COPY detection.py ./
COPY run.sh ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make the run script executable
RUN chmod +x run.sh

# Expose the add-on service entry point
CMD ["./run.sh"]
