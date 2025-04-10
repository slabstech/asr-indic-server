# ASR Indic Server

## Overview
Automatic Speech Recognition (ASR) for Indian languages using IndicConformer models. The default model is set to Kannada ASR.

## Demo Video

Watch a quick demo of our project in action! Click the image below to view the video on YouTube.

<a href="https://youtu.be/F0Mo0zjyysM" target="_blank">
  <img src="https://img.youtube.com/vi/F0Mo0zjyysM/0.jpg" alt="Watch the video">
</a>

## Table of Contents
- [Supported Languages](#supported-languages)
- [Live Server](#live-server)
- [Getting Started](#getting-started-development)
  - [For Production (Docker)](#for-production-docker)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [For Development (Local)](#for-development-local)
    - [Prerequisites](#prerequisites-1)
    - [Steps](#steps-1)
- [Downloading Translation Models](#downloading-translation-models)
  - [Kannada](#kannada)
  - [Other Languages](#other-languages)
    - [Malayalam](#malayalam)
    - [Hindi](#hindi)
- [Running with FastAPI Server](#running-with-fastapi-server)
- [Live Server](#live-server)
  - [Service Modes](#service-modes)
    - [High Latency, Slow System (Available 24/7)](#high-latency-slow-system-available-247)
    - [Low Latency, Fast System (Available on Request)](#low-latency-fast-system-available-on-request)
  - [How to Use the Service](#how-to-use-the-service)
    - [High Latency Service](#high-latency-service)
    - [Low Latency Service](#low-latency-service)
    - [Notes](#notes)
- [Evaluating Results](#evaluating-results)
  - [Kannada Transcription Examples](#kannada-transcription-examples)
    - [Sample 1: kannada_sample_1.wav](#sample-1-kannada_sample_1wav)
    - [Sample 2: kannada_sample_2.wav](#sample-2-kannada_sample_2wav)
    - [Sample 3 - Song - 4 minutes](#sample-3---song---4-minutes)
    - [Sample 4 - Song - 6.4 minutes](#sample-4---song---64-minutes)
  - [Batch Transcription Examples](#batch-transcription-examples)
    - [Transcribe Batch Endpoint](#transcribe-batch-endpoint)
- [Building Docker Image](#building-docker-image)
  - [Run the Docker Image](#run-the-docker-image)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Additional Resources](#additional-resources)
  - [Running Nemo Model](#running-nemo-model)
  - [Running with Transformers](#running-with-transformers)

## Supported Languages

22 Indian languages are supported, thanks to AIBharat organisation

| Language      | Code  |
|---------------|-------|
| Assamese      | `as`  |
| Bengali       | `bn`  |
| Bodo          | `brx` |
| Dogri         | `doi` |
| Gujarati      | `gu`  |
| Hindi         | `hi`  |
| Kannada       | `kn`  |
| Kashmiri      | `ks`  |
| Konkani       | `kok` |
| Maithili      | `mai` |
| Malayalam     | `ml`  |
| Manipuri      | `mni` |
| Marathi       | `mr`  |
| Nepali        | `ne`  |
| Odia          | `or`  |
| Punjabi       | `pa`  |
| Sanskrit      | `sa`  |
| Santali       | `sat` |
| Sindhi        | `sd`  |
| Tamil         | `ta`  |
| Telugu        | `te`  |
| Urdu          | `ur`  |

### Live Server

We have hosted an Automatic Speech Recognition (ASR) service that can be used to verify the accuracy of audio transcriptions. 

#### 
- [CPU - API Endpoint](https://huggingface.co/spaces/gaganyatri/asr_indic_server_cpu)

- [Via Gradio UI](https://huggingface.co/spaces/gaganyatri/asr_indic_app_gradio)

### Notes

- Ensure that the audio file path (`samples/kannada_sample_2.wav`) is correct and accessible.
- The `language` parameter in the URL specifies the language of the audio file. In the examples above, it is set to `kannada`.
- The service expects the audio file to be in WAV format.

## Getting Started - Development

### For Development (Local)
- **Prerequisites**: Python 3.10 (compatibility verified)
- **Steps**:
  1. **Create a virtual environment**:
  ```bash
  python -m venv venv
  ```
  2. **Activate the virtual environment**:
  ```bash
  source venv/bin/activate
  ```
  On Windows, use:
  ```bash
  venv\Scripts\activate
  ```
  3. **Install dependencies**:
      ```bash
      pip install -r requirements.txt
      ```
      - ### For Individual language models
        ```bash
        pip install -r nemo-requirements.txt
        ``` 

## Downloading Translation Models
Models can be downloaded from AI4Bharat's HuggingFace repository:

### For Multi-lingual language supported model
```bash
huggingface-cli download ai4bharat/indic-conformer-600m-multilingual
```

### For Individual langauge models 
-  Kannada
  ```bash
  huggingface-cli download ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large
  ```

### Other Languages
- [ASR  - IndicConformer Collection on HuggingFace](https://huggingface.co/collections/ai4bharat/indicconformer-66d9e933a243cba4b679cb7f)

  - Malayalam
  ```bash
  huggingface-cli download ai4bharat/indicconformer_stt_ml_hybrid_rnnt_large
  ```

  - Hindi
  ```bash
  huggingface-cli download ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large
  ```


### Sample Code
### For all languages
```python
from transformers import AutoModel
import torchaudio
import torch

# Load the model
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

# Load an audio file
wav, sr = torchaudio.load("kannada_sample_1.wav")
wav = torch.mean(wav, dim=0, keepdim=True)

target_sample_rate = 16000  # Expected sample rate
if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
    wav = resampler(wav)

# Perform ASR with CTC decoding
transcription_ctc = model(wav, "kn", "ctc")
print("CTC Transcription:", transcription_ctc)

# Perform ASR with RNNT decoding
transcription_rnnt = model(wav, "kn", "rnnt")
print("RNNT Transcription:", transcription_rnnt)

```

- Run the Code
  ```bash
  python asr-multi-lingual.py
  ```

### Individual Languages
```python
import torch
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.freeze() # inference mode
model = model.to(device)

model.cur_decoder = "rnnt"
rnnt_text = model.transcribe(['samples/kannada_sample_1.wav'], batch_size=1, language_id='kn')[0]


print(rnnt_text)
```


- Run the Code
  ```bash
  python asr_code.py
  ```


### Alternative examples for Development

#### For Local Development
-  Gradio 
```bash
python src/ux/app_local.py
```

#### For Server Development

#### Running with FastAPI Server
Run the server using FastAPI with the desired language (e.g., Kannada):
- for GPU
  ```bash
  python src/multi-lingual/asr_api.py --port 7860 --language kn --host 0.0.0.0 --device gpu
  ```
- for CPU only
  ```bash
  python src/multi-lingual/asr_api.py --port 7860 --language kn --host 0.0.0.0 --device cpu
  ```

#### Evaluating Results for FastApi Server
You can evaluate the ASR transcription results using `curl` commands. 
### Kannada Transcription Examples

#### Sample 1: kannada_sample_1.wav
- **Audio File**: [samples/kannada_sample_1.wav](samples/kannada_sample_1.wav)
- **Command**:
```bash
curl -X 'POST' 'http://loca?language=kannada' -H 'accept: application/json'   -H 'Content-Type: multipa'Content-Type  multipart/form-data' -F 'file=@samples/kannada_sample_1.wav;type=audio/x-wav'
```
- **Expected Output**:
```ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು```
Translation: "What is the capital of Karnataka"

#### Sample 2: kannada_sample_2.wav
- **Audio File**: [samples/kannada_sample_2.wav](samples/kannada_sample_2.wav)
- **Command**:
```bash
curl -X 'POST' \
'http://localhost:7860/transcribe/?language=kannada' \
-H 'accept: application/json'   -H 'Content-Type: multipart/form-data' \
-F 'file=@samples/kannada_sample_2.wav;type=audio/x-wav'
```
- **Expected Output**:
```ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿದೆ ಕರ್ನಾಟಕದಲ್ಲಿ ನಾವು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇವೆ```

#### Sample 3 - Song - 4 minutes
- [YT Video- Navaduva Nudiye](https://www.youtube.com/watch?v=LuZzhMN8ndQ)
- **Audio File**: [samples/kannada_sample_3.wav](samples/kannada_sample_3.wav)
- **Command**:
```bash
curl -X 'POST' \
'http://localhost:7860/transcribe/language=kannada' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'file=@samples/kannada_sample_3.wav;type=audio/x-wav'
```
- **Expected Output**: [kannada_sample_3_out.md](docs/kannada_sample_3_out.md)

#### Sample 4 - Song - 6.4 minutes
- [YT Video- Aagadu Yendu](https://www.youtube.com/watch?v=-Oryie1c-gs)
- **Audio File**: [samples/kannada_sample_4.wav](samples/kannada_sample_4.wav)
- **Command**:
```bash
curl -X 'POST' \
'http://localhost:7860/transcribe/language=kannada' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'file=@samples/kannada_sample_4.wav;type=audio/x-wav'
```
- **Expected Output**: [kannada_sample_4_out.md](docs/kannada_sample_4_out.md)

**Note**: The ASR does not provide sentence breaks or punctuation (e.g., question marks). We plan to integrate an LLM parser for improved context in future updates.

### Batch Transcription Examples

#### Transcribe Batch Endpoint
The `/transcribe_batch` endpoint allows you to transcribe multiple audio files in a single request. This is useful for batch processing of audio files.

- **Command**:
```bash
curl -X 'POST' \
'http://localhost:7860/transcribe_batch/' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'files=@samples/kannada_sample_1.wav;type=audio/x-wav' \
-F 'files=@samples/kannada_sample_2.wav;type=audio/x-wav'
```
- **Expected Output**:
```json
{
  "transcriptions": [
    "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು",
    "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿದೆ ಕರ್ನಾಟಕದಲ್ಲಿ ನಾವು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇವೆ"
  ]
}
```

## Troubleshooting
- **Transcription errors**: Verify the audio file is in WAV format, mono, and sampled at 16kHz. Adjust using:
```bash
ffmpeg -i sample_audio.wav -ac 1 -ar 16000 sample_audio_infer_ready.wav -y
```
- **Model not found**: Download the required models using the `huggingface-cli download` commands above.
- **Port conflicts**: Ensure port 7860 is free when running the FastAPI server.


## Contributing

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

Also you can join the [discord group](https://discord.gg/WZMCerEZ2P) to collaborate


## References
- [AI4Bharat IndicConformerASR GitHub Repository](https://github.com/AI4Bharat/IndicConformerASR)
- [Nemo - AI4Bharat](https://github.com/AI4Bharat/NeMo)
- [IndicConformer Collection on HuggingFace](https://huggingface.co/collections/ai4bharat/indicconformer-66d9e933a243cba4b679cb7f)

### Additional methods for Development

#### Running Nemo Model
1. Download the Nemo model:
```bash
wget https://objectstore.e2enetworks.net/indic-asr-public/indicConformer/ai4b_indicConformer_kn.nemo -O kannada.nemo
```
2. Adjust the audio:
```bash
ffmpeg -i sample_audio.wav -ac 1 -ar 16000 sample_audio_infer_ready.wav -y
```
3. Run the program:
```bash
python nemo_asr.py
```


<!-- 


- server-setup.sh - Use for container deployment on OlaKrutrim AI Pod
#### Building Docker Image
Build the Docker image locally:
```bash
docker build -t slabstech/asr_indic_server -f Dockerfile .
```

### Run the Docker Image
```
docker run --gpus all -it --rm -p 7860:7860 slabstech/asr_indic_server
```
- **Docker fails to start**: Ensure Docker is running and the `compose.yaml` file is correctly formatted.

-->



<!-- 
### For Production (Docker)
- **Prerequisites**: Docker and Docker Compose
- **Steps**:
  1. **Start the server**:
  For GPU
  ```bash
  docker compose -f compose.yaml up -d
  ```
  For CPU only
  ```bash
  docker compose -f cpu-compose.yaml up -d
  ```
  2. **Update source and target languages**:
  Modify the `compose.yaml` file to set the desired language. Example configurations:
  - **Kannada**:
  ```yaml
  language: kn
  ```
  - **Hindi**:
  ```yaml
  language: hi
  ```
-->
<!-- 
#### GPU / Paused, On-demand, $.05 /hour

```sh curl_low_latency.sh
curl -X 'POST' \
  'https://gaganyatri-asr-indic-server.hf.space/transcribe/?language=kannada' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@samples/kannada_sample_2.wav;type=audio/x-wav'
```
-->