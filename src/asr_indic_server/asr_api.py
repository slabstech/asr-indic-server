import torch
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment
import os
import tempfile
import subprocess

app = FastAPI()

# Load and prepare the model
model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.freeze() # inference mode
model = model.to(device) # transfer model to device

# Define the response model
class TranscriptionResponse(BaseModel):
    text: str

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Check file extension
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["wav", "mp3"]:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a WAV or MP3 file.")

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        # Convert MP3 to WAV if necessary
        if file_extension == "mp3":
            audio = AudioSegment.from_mp3(tmp_file_path)
            wav_file_path = tmp_file_path.replace(".mp3", ".wav")
            audio.export(wav_file_path, format="wav")
            os.remove(tmp_file_path)
            tmp_file_path = wav_file_path

        # Check the sample rate of the WAV file
        audio = AudioSegment.from_wav(tmp_file_path)
        sample_rate = audio.frame_rate

        # Convert WAV to the required format using ffmpeg if necessary
        if sample_rate != 16000:
            converted_wav_path = tmp_file_path.replace(".wav", "_infer_ready.wav")
            subprocess.run(
                ["ffmpeg", "-i", tmp_file_path, "-ac", "1", "-ar", "16000", converted_wav_path, "-y"],
                check=True
            )
            os.remove(tmp_file_path)
            tmp_file_path = converted_wav_path

        # Transcribe the audio
        model.cur_decoder = "rnnt"
        rnnt_text = model.transcribe([tmp_file_path], batch_size=1, language_id='kn')[0]

        # Clean up temporary file
        os.remove(tmp_file_path)

        return JSONResponse(content={"text": rnnt_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the server, use the following command:
# uvicorn src.asr_indic_server.asr:app --reload