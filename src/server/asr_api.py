
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.responses import JSONResponse
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
from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File, Form

from logging_config import logger

from asr import ASRManager

asr_manager = ASRManager()

app = FastAPI()

# Define the response model
class TranscriptionResponse(BaseModel):
    text: str

class BatchTranscriptionResponse(BaseModel):
    transcriptions: List[str]

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/v1/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Query(..., enum=list(asr_manager.model_language.keys())),
    #api_key: str = Depends(get_api_key),
    request: Request = None,  # Add for debugging
):
    logger.info(f"Request method: {request.method}, Headers: {request.headers}, Query: {request.query_params}")
    start_time = time()
    try:
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["wav", "mp3"]:
            logger.warning(f"Unsupported file format: {file_extension}")
            raise HTTPException(
                status_code=400, detail="Unsupported file format. Please upload a WAV or MP3 file."
            )

        file_content = await file.read()
        audio = (
            AudioSegment.from_mp3(io.BytesIO(file_content))
            if file_extension == "mp3"
            else AudioSegment.from_wav(io.BytesIO(file_content))
        )
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000).set_channels(1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            audio.export(tmp_file.name, format="wav")
            tmp_file_path = tmp_file.name

        chunk_file_paths = asr_manager.split_audio(tmp_file_path)
        try:
            language_id = asr_manager.model_language.get(language, asr_manager.default_language)
            transcription = asr_manager.transcribe(chunk_file_paths, language_id)
            logger.info(f"Transcription completed in {time() - start_time:.2f} seconds")
            return JSONResponse(content={"text": transcription})
        finally:
            for chunk_file_path in chunk_file_paths:
                if os.path.exists(chunk_file_path):
                    os.remove(chunk_file_path)
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            asr_manager.cleanup()
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/v1/transcribe_batch/", response_model=BatchTranscriptionResponse)
async def transcribe_audio_batch(
    files: List[UploadFile] = File(...),
    language: str = Query(..., enum=list(asr_manager.model_language.keys())),
    #api_key: str = Depends(get_api_key),
):
    start_time = time()
    all_transcriptions = []
    try:
        for file in files:
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ["wav", "mp3"]:
                logger.warning(f"Unsupported file format: {file_extension}")
                raise HTTPException(
                    status_code=400, detail="Unsupported file format. Please upload WAV or MP3 files."
                )

            file_content = await file.read()
            audio = (
                AudioSegment.from_mp3(io.BytesIO(file_content))
                if file_extension == "mp3"
                else AudioSegment.from_wav(io.BytesIO(file_content))
            )
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000).set_channels(1)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                audio.export(tmp_file.name, format="wav")
                tmp_file_path = tmp_file.name

            chunk_file_paths = asr_manager.split_audio(tmp_file_path)
            try:
                language_id = asr_manager.model_language.get(language, asr_manager.default_language)
                transcription = asr_manager.transcribe(chunk_file_paths, language_id)
                all_transcriptions.append(transcription)
            finally:
                for chunk_file_path in chunk_file_paths:
                    if os.path.exists(chunk_file_path):
                        os.remove(chunk_file_path)
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                asr_manager.cleanup()

        logger.info(f"Batch transcription completed in {time() - start_time:.2f} seconds")
        return JSONResponse(content={"transcriptions": all_transcriptions})
    except Exception as e:
        logger.error(f"Error during batch transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch transcription failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server for ASR.")
    parser.add_argument("--port", type=int, default=8888, help="Port to run the server on.")
    parser.add_argument("--language", type=str, default="kn", help="Default language for the ASR model.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)