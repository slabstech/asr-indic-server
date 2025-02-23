model_path = "kannada.nemo"
lang_id = "kn"

import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
model.eval() # inference mode
model = model.to(device)

model.cur_decoder = "ctc"
ctc_text = model.transcribe(['kannada_query_infer.wav'], batch_size=1, logprobs=False, language_id=lang_id)[0]
print(ctc_text)

model.cur_decoder = "rnnt"
ctc_text = model.transcribe(['kannada_query_infer.wav'], batch_size=1, logprobs=False, language_id=lang_id)[0]
print(ctc_text)
