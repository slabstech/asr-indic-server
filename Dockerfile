FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
# hadolint ignore=DL3008,DL3015,DL4006
RUN apt-get update && \
    apt-get install -y git curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.12 python3-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /root/asr_indic_server

COPY ./requirements.txt .
RUN pip3.12 install --no-cache-dir -r requirements.txt

COPY ./src ./src
ENV PYTHONPATH=/root/asr_indic_server/src
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
CMD ["uvicorn", "asr_indic_server.asr_api:app"]
