import torch
import nemo.collections.asr as nemo_asr
import time
import argparse

def load_model(model_name, device):
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model.freeze()  # inference mode
    model = model.to(device)  # transfer model to device
    return model

def transcribe_audio(model, audio_file, batch_size, language_id, decoder_type):
    model.cur_decoder = decoder_type
    transcribed_text = model.transcribe([audio_file], batch_size=batch_size, language_id=language_id)[0]
    return transcribed_text

def measure_execution_time(model, audio_file, batch_size, language_id, decoder_type):
    start_time = time.time()
    transcribed_text = transcribe_audio(model, audio_file, batch_size, language_id, decoder_type)
    end_time = time.time()
    execution_time = end_time - start_time
    return transcribed_text, execution_time

def main(device_type):
    model_name = "ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large"
    audio_file = 'kannada_query_infer.wav'
    batch_size = 1
    language_id = 'kn'
    decoder_type = "rnnt"

    device = torch.device(device_type if torch.cuda.is_available() and device_type == "cuda" else "cpu")
    model = load_model(model_name, device)
    transcribed_text, execution_time = measure_execution_time(model, audio_file, batch_size, language_id, decoder_type)

    print(f"Execution time on {device_type}: {execution_time:.4f} seconds")
    print(f"Transcribed text: {transcribed_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using ASR model.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device type to use for inference (cpu or cuda).")
    args = parser.parse_args()
    main(args.device)