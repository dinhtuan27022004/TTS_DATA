"""Test nhanh F5-TTS synthesize một câu."""

import os
import sys
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.tts.F5_V0 import F5TTSVietnamese

model = F5TTSVietnamese(
    vocoder_name="vocos",
    speed=1.0,
    ckpt_file = "/home/reg/TTS_DATA/models/f5-tts-v0/model.pt"
)

audio, sr = model.synthesize(
    gen_text="chất lượng được bảo chứng là thế, thế giới mới thì không phải ai cũng đọc được. Nhưng quan trọng là nội dung khá là hay , khiến cho nhiều người có thể hiểu được về thế giới này",
    ref_text= "chất lượng được bảo chứng là thế, nhưng mà đây không phải là một quyển sách nổi tiếng tại thị trường việt nam, phần vì nội dung khá là khó tiếp cận với thị yếu của chung của bạn đọc, phần vì bản thân sách không được quảng bá rầm rộ.", 
    ref_audio_path="/home/reg/TTS_DATA/models/samples/update_70000_gen.wav"
)

output_path = "/home/reg/TTS_DATA/data/test_output.wav"
sf.write(output_path, audio, sr)
print(f"Audio saved: {output_path} (sr={sr}, samples={len(audio)})")
