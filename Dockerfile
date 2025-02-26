# Use the official PyTorch image with CUDA support as base
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

USER user

ENV PATH="/home/user/.local/bin:$PATH"
# set work directory
WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
#RUN apt-get update && apt-get install -y netcat

# install dependencies
RUN pip install --upgrade pip
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt


# copy project

COPY --chown=user . /app

# CMD to run the application
CMD ["python", "src/asr_api.py", "--host", "0.0.0.0", "--port", "7860"]