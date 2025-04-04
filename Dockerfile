FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    sudo \
    wget \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY server-requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

#RUN pip install --no-cache-dir -r server-requirements.txt

COPY . .

RUN useradd -ms /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

# Use absolute path for clarity
CMD ["python", "/app/src/asr_api.py", "--host", "0.0.0.0", "--port", "7860", "--device", "cuda"]