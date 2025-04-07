import onnx
import onnxruntime
import numpy as np
import librosa
from scipy.io import wavfile

# Function to preprocess audio file and extract features
def preprocess_audio(audio_path, sample_rate=16000, max_length=10.0, n_mels=80):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    audio, _ = librosa.effects.trim(audio)
    max_samples = int(max_length * sample_rate)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    audio = audio / np.max(np.abs(audio))
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=160,
        win_length=400,
        fmax=8000
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    features = log_mel_spec.T  # [time_steps, n_mels]
    features = features.T      # [n_mels, time_steps]
    features = np.expand_dims(features, axis=0)  # [1, n_mels, time_steps]
    
    print(f"Audio features shape: {features.shape}")
    return features.astype(np.float32)

# Function to load tokens
def load_tokens(token_file):
    tokens = {}
    with open(token_file, 'r') as f:
        for line in f:
            token, idx = line.strip().split()
            tokens[int(idx)] = token
    return tokens

# Main inference class
class RNNTInference:
    def __init__(self, onnx_model_path, tokens_path):
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.tokens = load_tokens(tokens_path)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print("Model expected inputs:")
        for inp in self.session.get_inputs():
            print(f"Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
        print("Model outputs:")
        for out in self.session.get_outputs():
            print(f"Name: {out.name}, Shape: {out.shape}, Type: {out.type}")
            
    def greedy_decode(self, logprobs):
        """Greedy decoding for RNN-T with debug info"""
        # logprobs shape: [batch_size, time_steps, vocab_size]
        predictions = np.argmax(logprobs, axis=-1)  # [batch_size, time_steps]
        predictions = predictions[0]  # [time_steps]
        
        # Print debug info
        print(f"Logprobs shape: {logprobs.shape}")
        print(f"Predictions (first 20): {predictions[:20]}")
        print(f"Max logprob sample: {np.max(logprobs[0, :10], axis=-1)}")  # Confidence check
        
        transcription = []
        prev_token = None
        
        for pred in predictions:
            current_token = self.tokens.get(pred, '')
            if current_token != '<blk>' and current_token != prev_token:
                transcription.append(current_token)
            prev_token = current_token
            
        return ''.join(transcription)

    def transcribe(self, audio_path):
        audio_features = preprocess_audio(audio_path)
        inputs = {
            self.input_names[0]: audio_features,
            self.input_names[1]: np.array([audio_features.shape[2]], dtype=np.int64)
        }
        outputs = self.session.run(self.output_names, inputs)
        logprobs = outputs[0]  # Log probabilities
        transcription = self.greedy_decode(logprobs)
        return transcription

def main():
    onnx_model_path = "indicconformer.onnx"
    tokens_path = "tokens.txt"
    audio_path = "samples/kannada_sample_2.wav"  # Replace with your audio file path
    
    transcriber = RNNTInference(onnx_model_path, tokens_path)
    try:
        transcription = transcriber.transcribe(audio_path)
        print("Transcription:", transcription)
    except Exception as e:
        print(f"Error during transcription: {str(e)}")

if __name__ == "__main__":
    main()