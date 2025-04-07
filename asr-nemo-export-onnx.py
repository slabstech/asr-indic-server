import torch
import nemo.collections.asr as nemo_asr

# Step 1: Load the pretrained NeMo ASR model
model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large")

# Step 2: Prepare the model for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.freeze()  # Set the model to inference mode
model = model.to(device)

# Step 3: Export the model to ONNX format
onnx_export_path = "indicconformer.onnx"
model.export(onnx_export_path)
print(f"Model has been exported to ONNX format at: {onnx_export_path}")

# ctc model
'''
with open('tokens.txt', 'w') as f:
  for i, s in enumerate(model.decoder.vocabulary):
    f.write(f"{s} {i}\n")
  f.write(f"<blk> {i+1}\n")
'''

#rnnt
tokens = model.cfg.labels
with open('tokens.txt', 'w') as f:
    for i, s in enumerate(tokens):
        f.write(f"{s} {i}\n")
    f.write(f"<blk> {len(tokens)}\n")