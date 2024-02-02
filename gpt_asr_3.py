import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import whisper
import pyaudio
import numpy as np
import audioop

# Load the Facebook model and tokenizer
facebook_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
facebook_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load the Whisper model
whisper_model = whisper.load_model("tiny")

def transcribe_with_facebook(audio_data):
    np_frames = np.frombuffer(audio_data, dtype=np.int16)
    input_values = facebook_tokenizer(np_frames, return_tensors="pt").input_values
    with torch.no_grad():
        logits = facebook_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return facebook_tokenizer.decode(predicted_ids[0])

def transcribe_with_whisper(audio_data):
    np_frames = np.frombuffer(audio_data, dtype=np.int16)
    mel = whisper.log_mel_spectrogram(np_frames).astype(np.float32)
    with torch.no_grad():
        result = whisper_model.decode(mel)
    return result.text

# PyAudio configuration
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_CHUNKS = 20

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open the microphone stream
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    start=False)

print("Start speaking...")

frames = []
silence_frames = 0
recording = False

# Choose the model for transcription
use_whisper = True

try:
    stream.start_stream()
    while True:
        if not stream.is_active():
            break
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = audioop.rms(data, 2)

            if recording:
                frames.append(data)
                if rms < SILENCE_THRESHOLD:
                    silence_frames += 1
                    if silence_frames > SILENCE_CHUNKS:
                        audio_data = b''.join(frames)
                        transcription = transcribe_with_whisper(audio_data) if use_whisper else transcribe_with_facebook(audio_data)
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
            pass

except KeyboardInterrupt:
    print("\nExiting...")

# Stop and close the stream and audio
stream.stop_stream()
stream.close()
audio.terminate()


# pip install git+https://github.com/openai/whisper.git
