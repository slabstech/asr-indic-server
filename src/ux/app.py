import gradio as gr
import os
import requests
import json
import logging

# Set up logging
logging.basicConfig(filename='execution.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping of user-friendly language names to language IDs
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

def get_endpoint(use_localhost, service_type):
    logging.info(f"Getting endpoint for service: {service_type}, use_localhost: {use_localhost}")
    if use_localhost:
        port_mapping = {
            "asr": 10860,
            "translate": 8860,
        }
        base_url = f'http://localhost:{port_mapping[service_type]}'
    else:
        base_url = f'https://gaganyatri-asr-indic-server-cpu.hf.space'
    logging.info(f"Endpoint for {service_type}: {base_url}")
    return base_url

def transcribe_audio(audio_path, use_localhost):
    logging.info(f"Transcribing audio from {audio_path}, use_localhost: {use_localhost}")
    base_url = get_endpoint(use_localhost, "asr")
    url = f'{base_url}/transcribe/?language=kannada'
    files = {'file': open(audio_path, 'rb')}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        transcription = response.json()
        logging.info(f"Transcription successful: {transcription}")
        return transcription.get('text', '')
    except requests.exceptions.RequestException as e:
        logging.error(f"Transcription failed: {e}")
        return ""

def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

# Create the Gradio interface
with gr.Blocks(title="Dhwani - Voice to Text Transcription") as demo:
    gr.Markdown("# Voice to Text Transcription")
    gr.Markdown("Record your voice or upload a WAV file and Transcript the voice")


    translate_src_language = gr.Dropdown(
        choices=list(language_mapping.keys()),
        label="Source Language - Fixed",
        value="Kannada",
        interactive=False
    )
    '''
    translate_tgt_language = gr.Dropdown(
        choices=list(language_mapping.keys()),
        label="Target Language",
        value="English"
    )
    '''
    audio_input = gr.Microphone(type="filepath", label="Record your voice")
    audio_upload = gr.File(type="filepath", file_types=[".wav"], label="Upload WAV file")
    audio_output = gr.Audio(type="filepath", label="Playback", interactive=False)
    transcription_output = gr.Textbox(label="Transcription Result", interactive=False)


    use_localhost_checkbox = gr.Checkbox(label="Use Localhost", value=False, interactive=False, visible=False)
    #resubmit_button = gr.Button(value="Resubmit Translation")

    def process_audio(audio_path, use_localhost):
        logging.info(f"Processing audio from {audio_path}, use_localhost: {use_localhost}")
        transcription = transcribe_audio(audio_path, use_localhost)
        return transcription

    audio_input.stop_recording(
        fn=process_audio,
        inputs=[audio_input,use_localhost_checkbox],
        outputs=transcription_output
    )

    audio_upload.upload(
        fn=process_audio,
        inputs=[audio_upload, use_localhost_checkbox],
        outputs=transcription_output
    )

# Launch the interface
demo.launch()