model_path = "kannada.nemo"
lang_id = "kn"

import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
model.eval() # inference mode
model = model.to(device)

'''
model.cur_decoder = "ctc"
ctc_text = model.transcribe(['kannada_query_infer.wav'], batch_size=1, logprobs=False, language_id=lang_id)[0]
print(ctc_text)
'''
model.cur_decoder = "rnnt"
ctc_text = model.transcribe(['kannada_query_infer.wav'], batch_size=1, logprobs=False, language_id=lang_id)[0]
print(ctc_text)


'''
import time

# Start timing for CTC decoder
start_time_ctc = time.time()

model.cur_decoder = "ctc"
ctc_text = model.transcribe(['kannada_query_infer.wav'], batch_size=1, logprobs=False, language_id=lang_id)[0]
print(ctc_text)

end_time_ctc = time.time()
ctc_duration = end_time_ctc - start_time_ctc
print(f"CTC transcription took {ctc_duration:.4f} seconds")

# Start timing for RNNT decoder
start_time_rnnt = time.time()

model.cur_decoder = "rnnt"
rnnt_text = model.transcribe(['kannada_query_infer.wav'], batch_size=1, logprobs=False, language_id=lang_id)[0]
print(rnnt_text)

end_time_rnnt = time.time()
rnnt_duration = end_time_rnnt - start_time_rnnt
print(f"RNNT transcription took {rnnt_duration:.4f} seconds")

# Calculate and print the speed difference
speed_difference = rnnt_duration - ctc_duration
print(f"Speed difference: {speed_difference:.4f} seconds")

'''