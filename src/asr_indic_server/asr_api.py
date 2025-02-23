import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
import soundfile as sf
import torchaudio
from typing import List
import argparse
import os
import uvicorn

# recommended to run this on a gpu with flash_attn installed
# don't set attn_implemetation if you don't have flash_attn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ARTPARK-IISc/whisper-medium-vaani-kannada"

# Function to initialize the model based on the source and target languages
def initialize_model(model_name):
    # Load tokenizer and feature extractor individually
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="Kannada", task="translate")
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load the model
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(DEVICE)

    return processor, model

app = FastAPI(
    title="ASR Indic Server",
    description="A FastAPI server for Automatic Speech Recognition (ASR) in Indic languages.",
    version="1.0.0",
    docs_url="/swagger",  # Custom URL for Swagger documentation
    redoc_url="/redoc",   # Custom URL for ReDoc documentation
    openapi_url="/openapi.json"  # Custom URL for OpenAPI JSON
)

class TranscriptionRequest(BaseModel):
    audio_file: UploadFile

class TranscriptionResponse(BaseModel):
    transcription: str

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(request: UploadFile):
    if not request:
        raise HTTPException(status_code=400, detail="Audio file is required")

    audio_data, sample_rate = sf.read(await request.read())

    # Ensure the audio is 16kHz (Whisper expects 16kHz audio)
    if sample_rate != 16000:
        audio_data = audio_data.astype('float32')
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_data = resampler(torch.tensor(audio_data).unsqueeze(0)).squeeze().numpy()

    # Use the processor to prepare the input features
    input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)

    # Generate transcription (disable gradient calculation during inference)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # Decode the generated IDs into human-readable text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return TranscriptionResponse(transcription=transcription)

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Translation Server")
    parser.add_argument("--model_name", type=str, default=os.getenv('model_name', 'ARTPARK-IISc/whisper-medium-vaani-kannada'), help="Model Name")
    return parser.parse_args()

# Run the server using Uvicorn
if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name

    # Initialize the model with the provided languages
    processor, model = initialize_model(model_name)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    # Access Swagger documentation at http://localhost:8000/swagger
    # Access ReDoc documentation at http://localhost:8000/redoc
    # Access OpenAPI JSON at http://localhost:8000/openapi.json