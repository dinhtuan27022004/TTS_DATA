"""Test nhanh F5-TTS synthesize một câu."""

import os
import sys
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.tts.F5_V0 import F5TTSVietnamese

model = F5TTSVietnamese(
    vocoder_name="vocos",
    speed=1.0,
)

audio, sr = model.synthesize(
    gen_text="chỉ bằng cách luôn lỗ lực thì cuối cùng bạn mới được đền đáp",
    ref_text= "chỉ bằng cách luôn lỗ lực thì cuối cùng bạn mới được đền đáp", 
    ref_audio_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Processed_DATA", "VIVOS", "VIVOSSPK01_R002.wav"
    )
)

output_path = "test_output.wav"
sf.write(output_path, audio, sr)
print(f"Audio saved: {output_path} (sr={sr}, samples={len(audio)})")
