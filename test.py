import torch
from transformers import pipeline

whisper  = pipeline("automatic-speech-recognition", "openai/whisper-tiny")
transcription = whisper('Temp Audio Files/audio.wav', chunk_length_s=30)
# print(transcription["text"][:500])
print(transcription["text"])