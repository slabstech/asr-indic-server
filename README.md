# ASR Indic Server

## Overview

Automatic Speech Recognition for Indian Languages

- Default model - Kannada AST
  - MODEL=ARTPARK-IISc/whisper-medium-vaani-kannada
  - MODEL=ARTPARK-IISc/whisper-small-vaani-kannada

## Running with Docker Compose

1. **Start the server:**
   ```bash
   docker compose -f compose.yaml up -d
   ```

2. **Update source and target languages:**
   Modify the `compose.yaml` file to set the MODEL (`MODEL`) language as per your requirements. Example configurations:
   - **Kannada:**
     ```yaml
     MODEL=ARTPARK-IISc/whisper-medium-vaani-kannada
     ```
   - **Hindi:**
     ```yaml
     MODEL=ARTPARK-IISc/whisper-medium-vaani-hindi
     ```

## Evaluating Results

You can evaluate the translation results using `curl` commands. Here are some examples:

### Kannada
```bash
curl -X POST "http://localhost:8000/translate" \
 -H "Content-Type: application/json" \
 -d '{
       "sentences": ["Hello, how are you?", "Good morning!"],
       "src_lang": "eng_Latn",
       "tgt_lang": "kan_Knda"
     }'
```

### Hindi
```bash
curl -X POST "http://localhost:8000/translate" \
 -H "Content-Type: application/json" \
 -d '{
       "sentences": ["ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?", "ಶುಭೋದಯ!"],
       "src_lang": "kan_Knda",
       "tgt_lang": "eng_Latn"
     }'
```

## Setting Up the Development Environment

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

## Downloading Translation Models

Models can be downloaded from ARTPARK-IISc's HuggingFace repository:

### Kannada
```bash
huggingface-cli download ARTPARK-IISc/whisper-medium-vaani-kannada
```

## Running with FastAPI Server

You can run the server using FastAPI:

```bash
uvicorn asr_indic_server/asr_api:app --host 0.0.0.0 --port 8000 --MODEL ARTPARK-IISc/whisper-medium-vaani-kannada
```

## Build Docker image
```bash 
  docker build -t slabstech/asr_indic_server -f Dockerfile .
```

## References

- [ARTPARK-IISC Vaani Model](https://huggingface.co/ARTPARK-IISc/whisper-medium-vaani-kannada)
- [Vaani Dataset](https://huggingface.co/datasets/ARTPARK-IISc/Vaani)
- [Vaani @ IISC](https://vaani.iisc.ac.in/)
---

This README provides a comprehensive guide to setting up and running the Indic Translate Server. For more details, refer to the linked resources.