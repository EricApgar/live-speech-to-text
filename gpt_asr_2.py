import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import pyaudio
import numpy as np
import audioop

# Load pretrained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# PyAudio configuration
CHUNK = 512  # Reduced buffer size
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_CHUNKS = 10  # Reduced the number of silent chunks to consider speech ended

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open the microphone stream
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE, input=True,
    frames_per_buffer=CHUNK, 
    start=False)  # Start stream in a non-blocking way

print("Start speaking...")

frames = []
silence_frames = 0
recording = False

try:
    stream.start_stream()  # Start the audio stream
    while True:
        if not stream.is_active():
            break
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)  # Non-blocking read
            rms = audioop.rms(data, 2)

            if recording:
                frames.append(data)
                if rms < SILENCE_THRESHOLD:
                    silence_frames += 1
                    if silence_frames > SILENCE_CHUNKS:
                        np_frames = np.frombuffer(b''.join(frames), dtype=np.int16)
                        input_values = tokenizer(np_frames, return_tensors="pt").input_values
                        with torch.no_grad():
                            logits = model(input_values).logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = tokenizer.decode(predicted_ids[0])
                        print(transcription)
                        if "stop" in transcription.lower():
                            break
                        frames = []
                        recording = False
                        silence_frames = 0
                else:
                    silence_frames = 0
            else:
                if rms > SILENCE_THRESHOLD:
                    recording = True
                    frames = [data]
        except IOError:
            pass  # Handle buffer overflow and other I/O errors

except KeyboardInterrupt:
    print("\nExiting...")

# Stop and close the stream and audio
stream.stop_stream()
stream.close()
audio.terminate()
