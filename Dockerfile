# Use the official PyTorch image with CUDA support as base
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
# Set working directory
WORKDIR /root/asr_indic_server

# Update and install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY ./requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./src ./src

# Set environment variables
ENV PYTHONPATH=/root/asr_indic_server/src
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=7860


CMD ["python","src/asr_indic_server/asr_api.py"]

# Run the application
#CMD ["uvicorn", "asr_indic_server.asr_api:app", "--host", "0.0.0.0", "--port", "8000"]