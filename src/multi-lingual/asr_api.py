from fastapi import FastAPI, UploadFile
import torch
import torchaudio
from transformers import AutoModel
import argparse
import uvicorn
from pydantic import BaseModel
from pydub import AudioSegment
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse
from typing import List

# Initialize the FastAPI app
app = FastAPI()
class TranscriptionResponse(BaseModel):
    text: str


class ASRModelManager:
    def __init__(self, device_type="cuda"):
        self.device_type = device_type
        self.model_language = {
            "kannada": "kn", "hindi": "hi", "malayalam": "ml", "assamese": "as", "bengali": "bn",
            "bodo": "brx", "dogri": "doi", "gujarati": "gu", "kashmiri": "ks", "konkani": "kok",
            "maithili": "mai", "manipuri": "mni", "marathi": "mr", "nepali": "ne", "odia": "or",
            "punjabi": "pa", "sanskrit": "sa", "santali": "sat", "sindhi": "sd", "tamil": "ta",
            "telugu": "te", "urdu": "ur"
        }


# Load the model
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

asr_manager = ASRModelManager()  # Load Kannada, Hindi, Tamil, Telugu, Malayalam

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...), language: str = Query(..., enum=list(asr_manager.model_language.keys()))):
    # Load the uploaded audio file
    wav, sr = torchaudio.load(file.file)
    wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample if necessary
    target_sample_rate = 16000  # Expected sample rate
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        wav = resampler(wav)

    # Perform ASR with CTC decoding
    #transcription_ctc = model(wav, "kn", "ctc")

    # Perform ASR with RNNT decoding
    transcription_rnnt = model(wav, "kn", "rnnt")

    return JSONResponse(content={"text": transcription_rnnt})
'''
    return {
        "CTC Transcription": transcription_ctc,
        "RNNT Transcription": transcription_rnnt
    }
'''


class BatchTranscriptionResponse(BaseModel):
    transcriptions: List[str]

'''
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
'''
@app.get("/")
async def home():
    return RedirectResponse(url="/docs")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server for ASR.")
    parser.add_argument("--port", type=int, default=8888, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    args = parser.parse_args()
    asr_manager = ASRModelManager(device_type=args.device)

    uvicorn.run(app, host=args.host, port=args.port)