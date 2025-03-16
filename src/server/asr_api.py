import torch
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment
import os
import tempfile
import subprocess
import asyncio
import io
import logging
from logging.handlers import RotatingFileHandler
from time import time
from typing import List
import argparse
import uvicorn
import shutil

# Configure logging with log rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("transcription_api.log", maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

class ASRModelManager:
    def __init__(self, languages_to_load=["kn", "hi", "ta", "te", "ml"], device_type="cuda"):
        self.device_type = device_type
        self.model_language = {
            "kannada": "kn", "hindi": "hi", "malayalam": "ml", "assamese": "as", "bengali": "bn",
            "bodo": "brx", "dogri": "doi", "gujarati": "gu", "kashmiri": "ks", "konkani": "kok",
            "maithili": "mai", "manipuri": "mni", "marathi": "mr", "nepali": "ne", "odia": "or",
            "punjabi": "pa", "sanskrit": "sa", "santali": "sat", "sindhi": "sd", "tamil": "ta",
            "telugu": "te", "urdu": "ur"
        }
        self.config_models = {
            "as": "ai4bharat/indicconformer_stt_as_hybrid_rnnt_large",
            "bn": "ai4bharat/indicconformer_stt_bn_hybrid_rnnt_large",
            "brx": "ai4bharat/indicconformer_stt_brx_hybrid_rnnt_large",
            "doi": "ai4bharat/indicconformer_stt_doi_hybrid_rnnt_large",
            "gu": "ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large",
            "hi": "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large",
            "kn": "ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large",
            "ks": "ai4bharat/indicconformer_stt_ks_hybrid_rnnt_large",
            "kok": "ai4bharat/indicconformer_stt_kok_hybrid_rnnt_large",
            "mai": "ai4bharat/indicconformer_stt_mai_hybrid_rnnt_large",
            "ml": "ai4bharat/indicconformer_stt_ml_hybrid_rnnt_large",
            "mni": "ai4bharat/indicconformer_stt_mni_hybrid_rnnt_large",
            "mr": "ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large",
            "ne": "ai4bharat/indicconformer_stt_ne_hybrid_rnnt_large",
            "or": "ai4bharat/indicconformer_stt_or_hybrid_rnnt_large",
            "pa": "ai4bharat/indicconformer_stt_pa_hybrid_rnnt_large",
            "sa": "ai4bharat/indicconformer_stt_sa_hybrid_rnnt_large",
            "sat": "ai4bharat/indicconformer_stt_sat_hybrid_rnnt_large",
            "sd": "ai4bharat/indicconformer_stt_sd_hybrid_rnnt_large",
            "ta": "ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large",
            "te": "ai4bharat/indicconformer_stt_te_hybrid_rnnt_large",
            "ur": "ai4bharat/indicconformer_stt_ur_hybrid_rnnt_large"
        }
        # Load models for specified languages on startup
        self.models = {}
        self.load_initial_models(languages_to_load)

    def load_initial_models(self, languages):
        device = torch.device(self.device_type if torch.cuda.is_available() and self.device_type == "cuda" else "cpu")
        logging.info(f"Loading models on device: {device}")
        for lang_id in languages:
            if lang_id not in self.config_models:
                logging.warning(f"No model available for language ID: {lang_id}. Skipping.")
                continue
            try:
                model_name = self.config_models[lang_id]
                logging.info(f"Loading model for {lang_id}: {model_name}")
                model = nemo_asr.models.ASRModel.from_pretrained(model_name)
                model.freeze()  # Set to inference mode
                model = model.to(device)
                self.models[lang_id] = model
                logging.info(f"Successfully loaded model for {lang_id}")
            except Exception as e:
                logging.error(f"Failed to load model for {lang_id}: {str(e)}")

    def get_model(self, language_id):
        if language_id not in self.models:
            logging.warning(f"Model for {language_id} not pre-loaded. Loading now...")
            model = self.load_model(language_id)
            self.models[language_id] = model
        return self.models[language_id]

    def load_model(self, language_id):
        model_name = self.config_models.get(language_id, self.config_models["kn"])
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        device = torch.device(self.device_type if torch.cuda.is_available() and self.device_type == "cuda" else "cpu")
        model.freeze()
        model = model.to(device)
        return model

    def split_audio(self, file_path, chunk_duration_ms=15000):
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        if duration_ms > chunk_duration_ms:
            num_chunks = (duration_ms + chunk_duration_ms - 1) // chunk_duration_ms
            chunks = [audio[i*chunk_duration_ms:min((i+1)*chunk_duration_ms, duration_ms)] for i in range(num_chunks)]
            output_dir = "audio_chunks"
            os.makedirs(output_dir, exist_ok=True)
            chunk_file_paths = []
            for i, chunk in enumerate(chunks):
                chunk_file_path = os.path.join(output_dir, f"chunk_{i}.wav")
                chunk.export(chunk_file_path, format="wav")
                chunk_file_paths.append(chunk_file_path)
            return chunk_file_paths
        else:
            return [file_path]

    def cleanup(self):
        output_dir = "audio_chunks"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

app = FastAPI()
asr_manager = ASRModelManager(languages_to_load=["kn", "hi", "ta", "te", "ml"])  # Load Kannada, Hindi, Tamil, Telugu, Malayalam

class TranscriptionResponse(BaseModel):
    text: str

class BatchTranscriptionResponse(BaseModel):
    transcriptions: List[str]

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...), language: str = Query(..., enum=list(asr_manager.model_language.keys()))):
    start_time = time()
    try:
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["wav", "mp3"]:
            logging.warning(f"Unsupported file format: {file_extension}")
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a WAV or MP3 file.")

        file_content = await file.read()
        if file_extension == "mp3":
            audio = AudioSegment.from_mp3(io.BytesIO(file_content))
        else:
            audio = AudioSegment.from_wav(io.BytesIO(file_content))

        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000).set_channels(1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            audio.export(tmp_file.name, format="wav")
            tmp_file_path = tmp_file.name

        chunk_file_paths = asr_manager.split_audio(tmp_file_path)

        try:
            language_id = asr_manager.model_language.get(language, "kn")
            model = asr_manager.get_model(language_id)
            model.cur_decoder = "rnnt"

            transcriptions = []
            for chunk_file_path in chunk_file_paths:
                rnnt_texts = model.transcribe([chunk_file_path], batch_size=1, language_id=language_id)[0]
                if isinstance(rnnt_texts, list) and len(rnnt_texts) > 0:
                    transcriptions.append(rnnt_texts[0])
                else:
                    transcriptions.append(rnnt_texts)

            joined_transcriptions = ' '.join(transcriptions)
            end_time = time()
            logging.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
            return JSONResponse(content={"text": joined_transcriptions})

        finally:
            for chunk_file_path in chunk_file_paths:
                if os.path.exists(chunk_file_path):
                    os.remove(chunk_file_path)
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            asr_manager.cleanup()

    except HTTPException as e:
        logging.error(f"HTTPException: {str(e)}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/transcribe_batch/", response_model=BatchTranscriptionResponse)
async def transcribe_audio_batch(files: List[UploadFile] = File(...), language: str = Query(..., enum=list(asr_manager.model_language.keys()))):
    start_time = time()
    all_transcriptions = []
    try:
        for file in files:
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ["wav", "mp3"]:
                logging.warning(f"Unsupported file format: {file_extension}")
                raise HTTPException(status_code=400, detail="Unsupported file format. Please upload WAV or MP3 files.")

            file_content = await file.read()
            if file_extension == "mp3":
                audio = AudioSegment.from_mp3(io.BytesIO(file_content))
            else:
                audio = AudioSegment.from_wav(io.BytesIO(file_content))

            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000).set_channels(1)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                audio.export(tmp_file.name, format="wav")
                tmp_file_path = tmp_file.name

            chunk_file_paths = asr_manager.split_audio(tmp_file_path)

            try:
                language_id = asr_manager.model_language.get(language, "kn")
                model = asr_manager.get_model(language_id)
                model.cur_decoder = "rnnt"

                transcriptions = []
                for chunk_file_path in chunk_file_paths:
                    rnnt_texts = model.transcribe([chunk_file_path], batch_size=1, language_id=language_id)[0]
                    if isinstance(rnnt_texts, list) and len(rnnt_texts) > 0:
                        transcriptions.append(rnnt_texts[0])
                    else:
                        transcriptions.append(rnnt_texts)

                joined_transcriptions = ' '.join(transcriptions)
                all_transcriptions.append(joined_transcriptions)

            finally:
                for chunk_file_path in chunk_file_paths:
                    if os.path.exists(chunk_file_path):
                        os.remove(chunk_file_path)
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                asr_manager.cleanup()

        end_time = time()
        logging.info(f"Batch transcription completed in {end_time - start_time:.2f} seconds")
        return JSONResponse(content={"transcriptions": all_transcriptions})

    except HTTPException as e:
        logging.error(f"HTTPException: {str(e)}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server for ASR.")
    parser.add_argument("--port", type=int, default=8888, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    args = parser.parse_args()

    asr_manager = ASRModelManager(languages_to_load=["kn", "hi", "ta", "te", "ml"], device_type=args.device)
    uvicorn.run(app, host=args.host, port=args.port)