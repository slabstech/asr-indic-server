from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import nemo.collections.asr as nemo_asr
import shutil
import os
from tempfile import NamedTemporaryFile
from typing import Dict
from pydantic import BaseModel
import uvicorn

# Dictionary mapping language codes to model names
LANGUAGE_MODELS = {
    "hi": "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large",
    "bn": "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large",
    "ta": "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large",
    # Add more languages and their corresponding models as needed
}


class TranscriptionResponse(BaseModel):
    text: str
    language: str


app = FastAPI(
    title="Indian Languages ASR API",
    description="API for automatic speech recognition in Indian languages",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for loaded models
model_cache = {}


def get_model(language: str):
    """
    Get or load the ASR model for the specified language
    """
    if language not in LANGUAGE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {language}. Supported languages are: {list(LANGUAGE_MODELS.keys())}",
        )

    if language not in model_cache:
        try:
            model = nemo_asr.models.ASRModel.from_pretrained(LANGUAGE_MODELS[language])
            model_cache[language] = model
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model for language {language}: {str(e)}",
            )

    return model_cache[language]


@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    language: str,
    file: UploadFile = File(...),
):
    """
    Transcribe audio file in the specified Indian language

    Parameters:
    - language: Language code (e.g., 'hi' for Hindi, 'bn' for Bengali)
    - file: Audio file in WAV format

    Returns:
    - Transcription text and language
    """
    # Validate file format
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    # Get the appropriate model
    model = get_model(language)

    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        try:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file.flush()

            # Perform transcription
            transcriptions = model.transcribe([temp_file.name])

            if not transcriptions or len(transcriptions) == 0:
                raise HTTPException(status_code=500, detail="Transcription failed")

            return TranscriptionResponse(text=transcriptions[0], language=language)

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error during transcription: {str(e)}"
            )
        finally:
            # Clean up temporary file
            os.unlink(temp_file.name)


@app.get("/languages/")
async def get_supported_languages() -> Dict[str, str]:
    """
    Get list of supported languages and their model names
    """
    return LANGUAGE_MODELS


@app.get("/health/")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
