import torch
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, File, UploadFile, HTTPException
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

# Configure logging with log rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("transcription_api.log", maxBytes=10*1024*1024, backupCount=5), # 10MB per file, keep 5 backup files
        logging.StreamHandler() # This will also print logs to the console
    ]
)

app = FastAPI()

# Load and prepare the model
try:
    model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.freeze() # inference mode
    model = model.to(device) # transfer model to device
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Define the response model
class TranscriptionResponse(BaseModel):
    text: str

class BatchTranscriptionResponse(BaseModel):
    transcriptions: List[str]

def split_audio(file_path, chunk_duration_ms=15000):
    """
    Splits an audio file into chunks of specified duration if the audio duration exceeds the chunk duration.

    :param file_path: Path to the audio file.
    :param chunk_duration_ms: Duration of each chunk in milliseconds (default is 15000 ms or 15 seconds).
    """
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Get the duration of the audio in milliseconds
    duration_ms = len(audio)

    # Check if the duration is more than the specified chunk duration
    if duration_ms > chunk_duration_ms:
        # Calculate the number of chunks needed
        num_chunks = duration_ms // chunk_duration_ms
        if duration_ms % chunk_duration_ms != 0:
            num_chunks += 1

        # Split the audio into chunks
        chunks = [audio[i*chunk_duration_ms:(i+1)*chunk_duration_ms] for i in range(num_chunks)]

        # Create a directory to save the chunks
        output_dir = "audio_chunks"
        os.makedirs(output_dir, exist_ok=True)

        # Export each chunk to separate files
        chunk_file_paths = []
        for i, chunk in enumerate(chunks):
            chunk_file_path = os.path.join(output_dir, f"chunk_{i}.wav")
            chunk.export(chunk_file_path, format="wav")
            chunk_file_paths.append(chunk_file_path)
            print(f"Chunk {i} exported successfully to {chunk_file_path}.")

        return chunk_file_paths
    else:
        return [file_path]

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    start_time = time()
    try:
        # Check file extension
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["wav", "mp3"]:
            logging.warning(f"Unsupported file format: {file_extension}")
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a WAV or MP3 file.")

        # Read the file content
        file_content = await file.read()

        # Convert MP3 to WAV if necessary
        if file_extension == "mp3":
            audio = AudioSegment.from_mp3(io.BytesIO(file_content))
        else:
            audio = AudioSegment.from_wav(io.BytesIO(file_content))

        # Check the sample rate of the WAV file
        sample_rate = audio.frame_rate

        # Convert WAV to the required format using ffmpeg if necessary
        if sample_rate != 16000:
            audio = audio.set_frame_rate(16000).set_channels(1)

        # Export the audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            audio.export(tmp_file.name, format="wav")
            tmp_file_path = tmp_file.name

        # Split the audio if necessary
        chunk_file_paths = split_audio(tmp_file_path)

        try:
            # Transcribe the audio
            model.cur_decoder = "rnnt"
            rnnt_texts = model.transcribe(chunk_file_paths, batch_size=1, language_id='kn')
            rnnt_text = " ".join(rnnt_texts)
            end_time = time()
            logging.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
            return JSONResponse(content={"text": rnnt_text})
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg conversion failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {str(e)}")
        except Exception as e:
            logging.error(f"An error occurred during processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
        finally:
            # Clean up temporary files
            for chunk_file_path in chunk_file_paths:
                if os.path.exists(chunk_file_path):
                    os.remove(chunk_file_path)
    except HTTPException as e:
        logging.error(f"HTTPException: {str(e)}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/transcribe_batch/", response_model=BatchTranscriptionResponse)
async def transcribe_audio_batch(files: List[UploadFile] = File(...)):
    start_time = time()
    tmp_file_paths = []
    transcriptions = []
    try:
        for file in files:
            # Check file extension
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ["wav", "mp3"]:
                logging.warning(f"Unsupported file format: {file_extension}")
                raise HTTPException(status_code=400, detail="Unsupported file format. Please upload WAV or MP3 files.")

            # Read the file content
            file_content = await file.read()

            # Convert MP3 to WAV if necessary
            if file_extension == "mp3":
                audio = AudioSegment.from_mp3(io.BytesIO(file_content))
            else:
                audio = AudioSegment.from_wav(io.BytesIO(file_content))

            # Check the sample rate of the WAV file
            sample_rate = audio.frame_rate

            # Convert WAV to the required format using ffmpeg if necessary
            if sample_rate != 16000:
                audio = audio.set_frame_rate(16000).set_channels(1)

            # Export the audio to a temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                audio.export(tmp_file.name, format="wav")
                tmp_file_path = tmp_file.name

            # Split the audio if necessary
            chunk_file_paths = split_audio(tmp_file_path)
            tmp_file_paths.extend(chunk_file_paths)

        logging.info(f"Temporary file paths: {tmp_file_paths}")
        try:
            # Transcribe the audio files in batch
            model.cur_decoder = "rnnt"
            rnnt_texts = model.transcribe(tmp_file_paths, batch_size=len(files), language_id='kn')
            logging.info(f"Raw transcriptions from model: {rnnt_texts}")
            end_time = time()
            logging.info(f"Transcription completed in {end_time - start_time:.2f} seconds")

            # Flatten the list of transcriptions
            transcriptions = [text for sublist in rnnt_texts for text in sublist]
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg conversion failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {str(e)}")
        except Exception as e:
            logging.error(f"An error occurred during processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
        finally:
            # Clean up temporary files
            for tmp_file_path in tmp_file_paths:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
    except HTTPException as e:
        logging.error(f"HTTPException: {str(e)}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    return JSONResponse(content={"transcriptions": transcriptions})

# To run the server, use the following command:
# uvicorn src.asr_indic_server.asr_api:app --reload