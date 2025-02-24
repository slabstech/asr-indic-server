# ASR Indic Server

## Overview

Automatic Speech Recognition (ASR) for Indian Languages using IndicConformer models. The default model is set to Kannada AST.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Running with Docker Compose](#running-with-docker-compose)
  - [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Evaluating Results](#evaluating-results)
- [Downloading Translation Models](#downloading-translation-models)
- [Running with FastAPI Server](#running-with-fastapi-server)
- [Building Docker Image](#building-docker-image)
- [References](#references)

## Getting Started

### Prerequisites

- Docker and Docker Compose (for containerized deployment)
- Python 3.6+ (for development environment)

### Running with Docker Compose

1. **Start the server:**

   ```bash
   docker compose -f compose.yaml up -d
   ```

2. **Update source and target languages:**

   Modify the `compose.yaml` file to set the language as per your requirements. Example configurations:

   - **Kannada:**

     ```yaml
     language=kannada
     ```

   - **Hindi:**

     ```yaml
     language=hindi
     ```

### Setting Up the Development Environment

1. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**

   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Evaluating Results

You can evaluate the translation results using `curl` commands. Here are some examples:

### Kannada

<button onclick="playAudio()">Play - kannada_sample_1.wav</button>
<audio id="audio" src="kannada_sample_1.wav" preload="auto"></audio>

<button onclick="playAudio()">Play - kannada_sample_2.wav </button>
<audio id="audio" src="kannada_sample_2.wav" preload="auto"></audio>



<script>
  function playAudio() {
    var audio = document.getElementById('audio');
    audio.play();
  }
</script>

## Transcribe Audio Files

To transcribe audio files, use the following `curl` commands:

### Kannada Sample 1


```bash
curl -X 'POST' \
  'http://localhost:8000/transcribe/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@kannada_sample_1.wav;type=audio/x-wav'
```

### Expected Response

The expected response for the given audio file should be:

```kannada
ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು
```

This response means "What is the capital of Karnataka" in Kannada.

### Kannada Sample 2

```bash
curl -X 'POST' \
  'http://localhost:8000/transcribe/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@kannada_sample_2.wav;type=audio/x-wav'
```

#### Expected Response for Kannada Sample 2

The expected response for the given audio file should be:

```kannada
ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿದೆ  ಕರ್ನಾಟಕದಲ್ಲಿ ನಾವು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇವೆ.
```

This response means "Bengaluru is the capital of Karnataka. We speak Kannada in Karnataka." in Kannada.


- Note : The ASR does not provide sentence breaks, question marks. We plan to pass this to an LLM parser for better context.

## Downloading Translation Models

Models can be downloaded from AI4Bharat's HuggingFace repository:

### Kannada

```bash
huggingface-cli download ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large
```

### Other Languages

- **Malayalam:**

  ```bash
  huggingface-cli download ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large
  ```

- **Hindi:**

  ```bash
  huggingface-cli download ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large
  ```

## Running with FastAPI Server

You can run the server using FastAPI:

```bash
uvicorn asr_indic_server/asr_api:app --host 0.0.0.0 --port 8000 --language kannada
```

## Building Docker Image

```bash
docker build -t slabstech/asr_indic_server -f Dockerfile .
```

## References

- [AI4Bharat IndicConformerASR GitHub Repository](https://github.com/AI4Bharat/IndicConformerASR)
- [Nemo Model - Kannada](https://objectstore.e2enetworks.net/indic-asr-public/indicConformer/ai4b_indicConformer_kn.nemo)
- [IndicConformer Collection on HuggingFace](https://huggingface.co/collections/ai4bharat/indicconformer-66d9e933a243cba4b679cb7f)

## Additional Resources

### Running Nemo Model

1. **Download the Nemo model:**

   ```bash
   wget https://objectstore.e2enetworks.net/indic-asr-public/indicConformer/ai4b_indicConformer_kn.nemo -O kannada.nemo
   ```

2. **Adjust the audio:**

   ```bash
   ffmpeg -i sample_audio.wav -ac 1 -ar 16000 sample_audio_infer_ready.wav -y
   ```

3. **Run the program:**

   ```bash
   python nemo_asr.py
   ```

### Running with Transformers

```bash
python hf_asr.py
```

---

This README provides a comprehensive guide to setting up and running the ASR Indic Server. For more details, refer to the linked resources.