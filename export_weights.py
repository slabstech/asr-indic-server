import torch
import nemo.collections.asr as nemo_asr

# Load the pretrained NeMo ASR model
model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large")

# Set device and inference mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Set the decoder to RNN-T
model.cur_decoder = "rnnt"

# Print configs for debugging and confirmation
print("Full preprocessor config:")
print(model.cfg.preprocessor)
print("Decoder config:")
print(model.cfg.decoder)
print("Joint config:")
print(model.cfg.joint)

# Extract key parameters
sample_rate = model.cfg.preprocessor.get("sample_rate", 16000)
hop_length = model.cfg.preprocessor.get("frame_hop", 160)
win_length = model.cfg.preprocessor.get("frame_length", 400)
n_mels = model.cfg.preprocessor.get("n_mels", None) or model.cfg.preprocessor.get("nfilt", 80) or 80
vocab_size = 5633  # From your tokens.txt
pred_dim = model.cfg.decoder.get("hidden_size", 640)
num_layers = model.cfg.decoder.get("num_layers", 2)
blank_id = 5632  # Last token in tokens.txt

print(f"Using: sample_rate={sample_rate}, hop_length={hop_length}, win_length={win_length}, n_mels={n_mels}, "
      f"vocab_size={vocab_size}, pred_dim={pred_dim}, num_layers={num_layers}, blank_id={blank_id}")

# Save decoder and joint weights
torch.save(model.decoder.state_dict(), "decoder_weights.pt")
torch.save(model.joint.state_dict(), "joint_weights.pt")
print("Weights saved to decoder_weights.pt and joint_weights.pt")

# Optional: Test PyTorch transcription for reference
audio_path = "samples/kannada_sample_1.wav"
rnnt_text = model.transcribe([audio_path], batch_size=1, language_id='kn')[0]
print("PyTorch RNNT transcription:", rnnt_text)