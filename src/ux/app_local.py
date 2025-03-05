import gradio as gr
import os
import requests
import json
import logging
import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import spaces
# Set up logging
logging.basicConfig(filename='execution.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping of user-friendly language names to language IDs
language_mapping = {
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "English": "eng_Latn",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic)": "kas_Arab",
    "Kashmiri (Devanagari)": "kas_Deva",
    "Konkani": "gom_Deva",
    "Malayalam": "mal_Mlym",
    "Manipuri (Bengali)": "mni_Beng",
    "Manipuri (Meitei)": "mni_Mtei",
    "Maithili": "mai_Deva",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi (Arabic)": "snd_Arab",
    "Sindhi (Devanagari)": "snd_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab"
}

model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.freeze() # inference mode
model = model.to(device)

def convert_to_mono(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_channels(1)  # Convert to mono
    mono_audio_path = audio_path.replace('.wav', '_mono.wav')
    audio.export(mono_audio_path, format="wav")
    return mono_audio_path

@spaces.GPU
def transcribe_audio(audio_path):
    logging.info(f"Transcribing audio from {audio_path}")

    try:
        mono_audio_path = convert_to_mono(audio_path)
        model.cur_decoder = "rnnt"
        transcription = model.transcribe([mono_audio_path], batch_size=1, language_id='kn')[0]

        print(transcription)

        logging.info(f"Transcription successful: {transcription}")
        return transcription
    except requests.exceptions.RequestException as e:
        logging.error(f"Transcription failed: {e}")
        return "Sorry, failed to Transcribe Audio"

# Create the Gradio interface
with gr.Blocks(title="Dhwani - Voice to Text Translation") as demo:
    gr.Markdown("# Voice to Text Translation")
    gr.Markdown("Record your voice or upload a WAV file and Translate it to your required Indian Language")

    translate_src_language = gr.Dropdown(
        choices=list(language_mapping.keys()),
        label="Source Language - Fixed",
        value="Kannada",
        interactive=False
    )

    audio_input = gr.Microphone(type="filepath", label="Record your voice")
    audio_upload = gr.File(type="filepath", file_types=[".wav"], label="Upload WAV file")
    audio_output = gr.Audio(type="filepath", label="Playback", interactive=False)
    transcription_output = gr.Textbox(label="Transcription Result", interactive=False)

    def process_audio(audio_path):
        logging.info(f"Processing audio from {audio_path}")
        transcription = transcribe_audio(audio_path)
        return transcription

    audio_input.stop_recording(
        fn=process_audio,
        inputs=[audio_input],
        outputs=transcription_output
    )

    audio_upload.upload(
        fn=process_audio,
        inputs=[audio_upload],
        outputs=transcription_output
    )

# Launch the interface
demo.launch()