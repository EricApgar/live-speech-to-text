import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import pyaudio
import wave
import numpy as np

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
DEVICE_INDEX = 0  # You might need to change this depending on your audio input device

# Load pretrained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Audio stream setup
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK, input_device_index=DEVICE_INDEX)

print("Recording...")

frames = []

def transcribe(frames):
    # Convert frames to audio format
    audio_buffer = np.frombuffer(b''.join(frames), dtype=np.int16)
    input_values = tokenizer(audio_buffer, return_tensors="pt").input_values

    # Predict and decode
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    return transcription

try:
    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        if len(frames) * CHUNK >= RATE:  # transcribe every 1 second
            transcription = transcribe(frames)
            print(transcription)
            frames = []

            if "stop" in transcription.lower():
                print("Stopping transcription.")
                break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    audio.terminate()
    print("Terminated")
