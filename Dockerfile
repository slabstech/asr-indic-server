# Use the official NVIDIA CUDA image as the base image
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip git ffmpeg sudo

# Set Python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create a non-root user
RUN useradd -ms /bin/bash appuser

# Switch to the non-root user
USER appuser

# Set the working directory for the non-root user
WORKDIR /home/appuser

# Expose the port the app runs on
EXPOSE 8888

# Command to run the application
CMD ["python", "src/asr_api.py", "--host", "0.0.0.0", "--port", "8888", "--device", "cuda"]