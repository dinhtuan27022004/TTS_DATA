# pyrefly: ignore [missing-import]
from datasets import load_from_disk
dataset= load_from_disk("/home/reg/TTS_DATA/data/infore1_25hours")

print(dataset)
# print(dataset[0]["transcription"])