import onnx
import onnxruntime
import numpy as np
import librosa
import torch
import torch.nn as nn

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
    reverse_tokens = {}
    with open(token_file, 'r') as f:
        for line in f:
            token, idx = line.strip().split()
            idx = int(idx)
            tokens[idx] = token
            reverse_tokens[token] = idx
    return tokens, reverse_tokens

# RNN-T Decoder
class RNNTDecoder:
    def __init__(self, vocab_size, enc_dim=512, pred_dim=640, num_layers=2, blank_id=5632, max_symbols=10):
        self.vocab_size = vocab_size
        self.enc_dim = enc_dim
        self.pred_dim = pred_dim
        self.num_layers = num_layers
        self.blank_id = blank_id
        self.max_symbols = max_symbols
        
        # Prediction network (LSTM)
        self.pred_rnn = nn.LSTM(input_size=vocab_size, hidden_size=pred_dim, num_layers=num_layers, batch_first=True)
        # Joint network
        self.joint_fc = nn.Linear(enc_dim + pred_dim, vocab_size)
        
        # Load trained weights
        try:
            self.pred_rnn.load_state_dict(torch.load("decoder_weights.pt"))
            self.joint_fc.load_state_dict(torch.load("joint_weights.pt"))
            print("Loaded trained weights for decoder and joint network")
        except FileNotFoundError as e:
            print(f"Error: Weight files not found. Please run export_weights.py first. {str(e)}")
            raise
        
        # For numpy conversion
        self.to_numpy = lambda x: x.detach().cpu().numpy()

    def predict(self, token_id, states):
        input_tensor = torch.zeros(1, 1, self.vocab_size, dtype=torch.float32)
        if token_id is not None:
            input_tensor[0, 0, token_id] = 1.0  # One-hot encoding
        
        with torch.no_grad():
            output, (h, c) = self.pred_rnn(input_tensor, states)
        return self.to_numpy(output[0, 0]), (self.to_numpy(h), self.to_numpy(c))

    def joint(self, enc_output, pred_output):
        enc_tensor = torch.tensor(enc_output, dtype=torch.float32).unsqueeze(0)  # [1, 512]
        pred_tensor = torch.tensor(pred_output, dtype=torch.float32).unsqueeze(0)  # [1, pred_dim]
        combined = torch.cat([enc_tensor, pred_tensor], dim=-1)  # [1, 512 + pred_dim]
        with torch.no_grad():
            logits = self.joint_fc(combined)  # [1, vocab_size]
        return self.to_numpy(logits[0])  # [vocab_size]

# Main inference class
class RNNTInference:
    def __init__(self, encoder_path, tokens_path, vocab_size=5633, enc_dim=512, pred_dim=640, num_layers=2, blank_id=5632):
        self.encoder_session = onnxruntime.InferenceSession(encoder_path)
        self.encoder_input_names = [input.name for input in self.encoder_session.get_inputs()]
        self.encoder_output_names = [output.name for output in self.encoder_session.get_outputs()]
        self.tokens, self.reverse_tokens = load_tokens(tokens_path)
        self.decoder = RNNTDecoder(vocab_size, enc_dim, pred_dim, num_layers, blank_id)
        
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
        enc_output = enc_outputs  # [1, 512, time_steps]
        enc_length = int(enc_lengths[0])
        print(f"enc_output shape: {enc_output.shape}, Length: {enc_length}")
        
        transcription = []
        hyp = [self.decoder.blank_id]  # Start with blank_id as SOS
        states = (np.zeros((self.decoder.num_layers, 1, self.decoder.pred_dim), dtype=np.float32),
                  np.zeros((self.decoder.num_layers, 1, self.decoder.pred_dim), dtype=np.float32))
        
        for t in range(enc_length):
            enc_t = enc_output[0, :, t]  # [512]
            not_blank = True
            symbols_added = 0
            
            while not_blank and symbols_added < self.decoder.max_symbols:
                pred_output, states = self.decoder.predict(hyp[-1], states)
                logits = self.decoder.joint(enc_t, pred_output)
                
                pred_idx = np.argmax(logits)
                if pred_idx == self.decoder.blank_id:
                    not_blank = False
                else:
                    hyp.append(pred_idx)
                    transcription.append(self.tokens.get(pred_idx, ''))
                symbols_added += 1
        
        return ''.join(transcription).replace('▁', ' ').strip()

    def transcribe(self, audio_path):
        audio_features = preprocess_audio(audio_path)
        enc_outputs, enc_lengths = self.encode(audio_features)
        transcription = self.greedy_decode(enc_outputs, enc_lengths)
        return transcription

def main():
    encoder_path = "encoder-indicconformer_rnnt_full.onnx"
    tokens_path = "tokens.txt"
    audio_path = "samples/kannada_sample_1.wav"
    
    # Initialize with parameters matching NeMo config (adjustable after export output)
    transcriber = RNNTInference(
        encoder_path=encoder_path,
        tokens_path=tokens_path,
        vocab_size=5633,
        enc_dim=512,  # From encoder output
        pred_dim=640,  # Default, adjust based on decoder config
        num_layers=2,  # Default, adjust based on decoder config
        blank_id=5632  # From tokens.txt
    )
    try:
        transcription = transcriber.transcribe(audio_path)
        print("Transcription:", transcription)
    except Exception as e:
        print(f"Error during transcription: {str(e)}")

if __name__ == "__main__":
    main()