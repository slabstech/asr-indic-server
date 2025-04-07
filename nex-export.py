import torch
import nemo.collections.asr as nemo_asr

# Load the pretrained NeMo ASR model
model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large")

# Set device and inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Use eval() for inference mode

# Set the decoder to RNN-T
model.cur_decoder = "rnnt"

# Print the full preprocessor and model config to debug
print("Full preprocessor config:")
print(model.cfg.preprocessor)
print("Full model config:")
print(model.cfg)

# Extract key parameters with fallbacks
sample_rate = model.cfg.preprocessor.get("sample_rate", 16000)
hop_length = model.cfg.preprocessor.get("frame_hop", 160)
win_length = model.cfg.preprocessor.get("frame_length", 400)
n_mels = model.cfg.preprocessor.get("n_mels", None) or model.cfg.preprocessor.get("nfilt", 80) or 80

# Vocab size fallback
if hasattr(model, "tokenizer"):
    vocab_size = len(model.tokenizer.vocab) + 1
elif "labels" in model.cfg.decoder:
    vocab_size = len(model.cfg.decoder.labels) + 1
else:
    vocab_size = 5633  # From previous tokens.txt
print(f"Using: sample_rate={sample_rate}, hop_length={hop_length}, win_length={win_length}, n_mels={n_mels}, vocab_size={vocab_size}")

# Define dummy inputs for encoder only
audio_length = 10.0  # Adjust based on your sample duration
time_steps = int(audio_length * sample_rate / hop_length)
dummy_audio = torch.randn(1, n_mels, time_steps).to(device)  # [1, n_mels, time_steps]
dummy_length = torch.tensor([time_steps], dtype=torch.int64).to(device)  # [1]

# Input example for encoder
input_example = (dummy_audio, dummy_length)

# Export to ONNX
onnx_path = "indicconformer_rnnt_full.onnx"
try:
    model.export(
        output=onnx_path,
        input_example=input_example,
        dynamic_axes={
            "audio_signal": {0: "batch_size", 2: "time_steps"},
            "length": {0: "batch_size"},
            "logprobs": {0: "batch_size", 1: "output_time_steps"}
        },
        verbose=True,
        do_constant_folding=True
    )
    print(f"Model exported to {onnx_path}")
except Exception as e:
    print(f"Export failed: {str(e)}")