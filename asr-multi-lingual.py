from transformers import AutoModel
import torchaudio
import torch

# Load the model
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

# Load an audio file
wav, sr = torchaudio.load("kannada_sample_1.wav")
wav = torch.mean(wav, dim=0, keepdim=True)

target_sample_rate = 16000  # Expected sample rate
if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
    wav = resampler(wav)

# Perform ASR with CTC decoding
transcription_ctc = model(wav, "kn", "ctc")
print("CTC Transcription:", transcription_ctc)

# Perform ASR with RNNT decoding
transcription_rnnt = model(wav, "kn", "rnnt")
print("RNNT Transcription:", transcription_rnnt)
