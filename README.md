# ASR Indic Server

## Overview

Automatic Speech Recognition for Indian Languages

- Default model - Kannada AST
  - language=kannada

## Running with Docker Compose

1. **Start the server:**
   ```bash
   docker compose -f compose.yaml up -d
   ```

2. **Update source and target languages:**
   Modify the `compose.yaml` file to set the language (`language`) language as per your requirements. Example configurations:
   - **Kannada:**
     ```yaml
     language=kannada
     ```
   - **Hindi:**
     ```yaml
     language=hindi
     ```

## Evaluating Results

You can evaluate the translation results using `curl` commands. Here are some examples:

### Kannada
```bash
curl -X 'POST' \
  'http://localhost:8000/transcribe/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@kannada.wav;type=audio/x-wav'
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

Models can be downloaded from AI4Bharats HuggingFace repository:

### Kannada
```bash
huggingface-cli download ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large
```

## Running with FastAPI Server

You can run the server using FastAPI:

```bash
uvicorn asr_indic_server/asr_api:app --host 0.0.0.0 --port 8000 --language kannada
```

## Build Docker image
```bash 
  docker build -t slabstech/asr_indic_server -f Dockerfile .
```

## References
  - https://github.com/AI4Bharat/IndicConformerASR

  - nemo model - kannada- https://objectstore.e2enetworks.net/indic-asr-public/indicConformer/ai4b_indicConformer_kn.nemo
---

This README provides a comprehensive guide to setting up and running the Indic Translate Server. For more details, refer to the linked resources.

-- Indic Conformer

 - IndicConformer Collection - https://huggingface.co/collections/ai4bharat/indicconformer-66d9e933a243cba4b679cb7f
  - Download models 
    - kannada - huggingface-cli download ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large
    - Malayalam - ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large
    - Hindi - ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large
 
- To run Nemo model >  nemo_asr.py 
  - Download the nemo model 
      - ```wget https://objectstore.e2enetworks.net/indic-asr-public/indicConformer/ai4b_indicConformer_kn.nemo -O kannada.nemo```

  - Adjust the audio
    - ```ffmpeg -i sample_audio.wav -ac 1 -ar 16000 sample_audio_infer_ready.wav -y```
  - Run the program
  - ```python nemo_asr.py```

- To run with Transformers > hf_asr.py
  - ``python hf_asr.py ```
  