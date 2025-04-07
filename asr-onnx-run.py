import onnxruntime as ort
import numpy as np
import soundfile as sf
import librosa

# Step 1: Load the ONNX model
onnx_model_path = "indicconformer.onnx"
session = ort.InferenceSession(onnx_model_path)

# Step 2: Preprocess the audio file
def preprocess_audio(audio_path, target_sr=16000, feature_dim=80):
    """
    Preprocesses the audio file by converting it to a log-Mel spectrogram.

    Args:
        audio_path (str): Path to the audio file.
        target_sr (int): Target sampling rate for the audio.
        feature_dim (int): Number of Mel filter banks (feature dimension).

    Returns:
        np.ndarray: Preprocessed log-Mel spectrogram with shape [1, sequence_length, feature_dim].
        np.ndarray: Array containing the sequence length.
    """
    # Load audio file
    audio, sr = sf.read(audio_path, dtype="float32")
    
    # Ensure the sample rate matches the target sample rate
    if sr != target_sr:
        raise ValueError(f"Audio sample rate must be {target_sr} Hz. Current sample rate: {sr}")
    
    # Convert raw audio to a log-Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=target_sr,
        n_mels=feature_dim,
        hop_length=512,
        n_fft=1024
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    
    # Transpose to match model input format and add batch dimension
    return np.expand_dims(log_mel_spectrogram.T, axis=0), np.array([log_mel_spectrogram.shape[1]], dtype=np.int64)

# Path to your audio file
audio_file = "samples/kannada_sample_1.wav"

# Preprocess the audio file
audio_data, audio_length = preprocess_audio(audio_file)

# Step 3: Load tokens from tokens.txt
def load_vocabulary(tokens_file):
    """
    Loads the vocabulary from a tokens.txt file.

    Args:
        tokens_file (str): Path to the tokens.txt file.

    Returns:
        dict: A dictionary mapping token indices to their corresponding text representations.
    """
    vocabulary = {}
    with open(tokens_file, "r") as f:
        for line in f:
            token, idx = line.strip().split()
            vocabulary[int(idx)] = token
    return vocabulary

# Path to your tokens file
tokens_file = "tokens.txt"

# Load vocabulary
vocabulary = load_vocabulary(tokens_file)

# Step 4: Perform inference
input_names = [input.name for input in session.get_inputs()]
output_name = session.get_outputs()[0].name

# Pass the audio data and length to the model
inputs_feed = {
    "audio_signal": audio_data.astype(np.float32),
    "length": audio_length,
}

# Perform inference
outputs = session.run([output_name], inputs_feed)[0]

# Step 5: Decode the results (using loaded vocabulary)
def decode_output(output_ids, vocabulary):
    """
    Decodes the output IDs into human-readable text using the vocabulary.

    Args:
        output_ids (np.ndarray): Array of output token IDs from the model.
        vocabulary (dict): A dictionary mapping token indices to text.

    Returns:
        str: Decoded transcription.
    """
    decoded_text = []
    for id in output_ids:
        if id in vocabulary:
            token = vocabulary[id]
            if token == "<space>":
                decoded_text.append(" ")  # Replace <space> with an actual space
            elif token != "<blk>":  # Ignore blank tokens
                decoded_text.append(token)
    return "".join(decoded_text)

# Decode and print transcribed text
transcribed_text = decode_output(outputs[0], vocabulary)
print(f"Transcribed text: {transcribed_text}")
