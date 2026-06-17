from datasets import load_from_disk

ds = load_from_disk("/home/reg/TTS_DATA/VieNeu-TTS/pnnbao-ump___vie_neu-tts/default/0.0.0/a5f8845053018f68467d45d5804b83711c7c1a01/vie_neu-tts-train-00000-of-00047.arrow")

for i in range(10):
    print(ds[i])