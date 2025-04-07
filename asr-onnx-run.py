

import onnxruntime as ort
import numpy as np
import soundfile as sf

# Step 1: Load the ONNX model
onnx_model_path = "indicconformer.onnx"
session = ort.InferenceSession(onnx_model_path)

# Step 2: Preprocess the audio file
def preprocess_audio(audio_path, target_sr=16000):
    audio, sr = sf.read(audio_path, dtype="float32")
    if sr != target_sr:
        raise ValueError(f"Audio sample rate must be {target_sr} Hz. Current sample rate: {sr}")
    return np.expand_dims(audio, axis=0)  # Add batch dimension

audio_file = "samples/kannada_sample_1.wav"
audio_data = preprocess_audio(audio_file)

# Step 3: Perform inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Pass the audio data to the model
outputs = session.run([output_name], {input_name: audio_data})[0]

# Step 4: Decode the results (assuming token IDs are mapped to text)
def decode_output(output_ids, vocabulary):
    return "".join([vocabulary[id] for id in output_ids if id < len(vocabulary)])

# Example vocabulary mapping (replace with actual tokens.txt content)
vocabulary = ["a", "b", "c", "<space>", "<blk>"]  # Example tokens; replace with your model's tokens.txt

transcribed_text = decode_output(outputs[0], vocabulary)
print(f"Transcribed text: {transcribed_text}")
