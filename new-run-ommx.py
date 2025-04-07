import onnx
import onnxruntime
import numpy as np
import librosa

# Function to preprocess audio
def preprocess_audio(audio_path, sample_rate=16000, n_mels=80, hop_length=160, win_length=400):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    print(f"Audio length: {len(audio)/sample_rate:.2f} seconds")
    
    audio = audio / np.max(np.abs(audio))
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        fmax=8000,
        power=1.0
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=1.0)
    
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

# Simple RNN-T Decoder
class SimpleRNNTDecoder:
    def __init__(self, vocab_size, enc_dim=512, pred_dim=320):
        self.vocab_size = vocab_size
        self.enc_dim = enc_dim  # Encoder output dimension
        self.pred_dim = pred_dim  # Prediction network dimension
        # Random weights for simulation
        self.pred_weight = np.random.randn(pred_dim, pred_dim).astype(np.float32)
        self.joint_weight = np.random.randn(enc_dim + pred_dim, vocab_size).astype(np.float32)

    def predict(self, prev_token, state):
        # Simulate prediction network
        token_embedding = np.zeros(self.pred_dim, dtype=np.float32)
        if prev_token is not None:
            token_embedding = np.array(prev_token, dtype=np.float32)[:self.pred_dim]
        new_state = np.tanh(np.dot(token_embedding, self.pred_weight))
        return new_state, new_state  # Simplified state update

    def joint(self, enc_output, pred_output):
        # Ensure inputs are 1D and concatenate
        enc_output = np.reshape(enc_output, (-1,))  # [512]
        pred_output = np.reshape(pred_output, (-1,))  # [320]
        combined = np.concatenate([enc_output, pred_output])  # [512 + 320]
        logits = np.dot(combined, self.joint_weight)  # [vocab_size]
        return logits

# Main inference class
class RNNTInference:
    def __init__(self, encoder_path, tokens_path, vocab_size=5633, enc_dim=512, pred_dim=320):
        self.encoder_session = onnxruntime.InferenceSession(encoder_path)
        self.encoder_input_names = [input.name for input in self.encoder_session.get_inputs()]
        self.encoder_output_names = [output.name for output in self.encoder_session.get_outputs()]
        self.tokens = load_tokens(tokens_path)
        self.decoder = SimpleRNNTDecoder(vocab_size, enc_dim, pred_dim)
        
        print("Encoder expected inputs:")
        for inp in self.encoder_session.get_inputs():
            print(f"Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
        print("Encoder outputs:")
        for out in self.encoder_session.get_outputs():
            print(f"Name: {out.name}, Shape: {out.shape}, Type: {out.type}")

    def encode(self, audio_features):
        inputs = {
            self.encoder_input_names[0]: audio_features,
            self.encoder_input_names[1]: np.array([audio_features.shape[2]], dtype=np.int64)
        }
        outputs = self.encoder_session.run(self.encoder_output_names, inputs)
        return outputs[0], outputs[1]  # [1, 512, time_steps], [batch_size]

    def greedy_decode(self, enc_outputs, enc_lengths):
        enc_output = enc_outputs[0]  # [1, 512, time_steps]
        enc_length = int(enc_lengths[0])  # e.g., 123
        
        transcription = []
        prev_token = None
        state = np.zeros(self.decoder.pred_dim, dtype=np.float32)
        
        for t in range(enc_length):
            enc_t = enc_output[0, :, t]  # [512] (transpose to match dims)
            pred_output, state = self.decoder.predict(prev_token, state)  # [320]
            logits = self.decoder.joint(enc_t, pred_output)  # [vocab_size]
            
            pred_idx = np.argmax(logits)
            current_token = self.tokens.get(pred_idx, '')
            if current_token != '<blk>' and current_token != prev_token:
                transcription.append(current_token)
            prev_token = current_token if current_token != '<blk>' else None
        
        return ''.join(transcription)

    def transcribe(self, audio_path):
        audio_features = preprocess_audio(audio_path)
        enc_outputs, enc_lengths = self.encode(audio_features)
        print(f"Encoded outputs shape: {enc_outputs.shape}, Lengths: {enc_lengths}")
        transcription = self.greedy_decode(enc_outputs, enc_lengths)
        return transcription

def main():
    encoder_path = "encoder-indicconformer_rnnt_full.onnx"
    tokens_path = "tokens.txt"
    audio_path = "samples/kannada_sample_1.wav"
    
    transcriber = RNNTInference(encoder_path, tokens_path)
    try:
        transcription = transcriber.transcribe(audio_path)
        print("Transcription:", transcription)
    except Exception as e:
        print(f"Error during transcription: {str(e)}")

if __name__ == "__main__":
    main()