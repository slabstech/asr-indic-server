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
from time import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Load and prepare the model
try:
    model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.freeze()  # inference mode
    model = model.to(device)  # transfer model to device
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Define the response model
class TranscriptionResponse(BaseModel):
    text: str

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

        try:
            # Transcribe the audio
            model.cur_decoder = "rnnt"
            rnnt_text = model.transcribe([tmp_file_path], batch_size=1, language_id='kn')[0]

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
            # Clean up temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    except HTTPException as e:
        logging.error(f"HTTPException: {str(e)}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# To run the server, use the following command:
# uvicorn src.asr_indic_server.asr_api:app --reload