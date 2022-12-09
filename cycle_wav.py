import time
from transformers import pipeline
import os
import pathlib
import pyaudio
import wave


repo_folder = os.path.dirname(str(pathlib.Path(__file__).absolute()))
audio_file = os.path.join(repo_folder, 'Temp Audio Files', 'audio.wav')

def main(stop_time: float=10) -> None:

    start_time = time.time()

    while (time.time() - start_time) < stop_time:

        record_sample(length=2)
        speech_to_text()

    return

def record_sample(length: float=1) -> None:

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  #2
    RATE = 16000  #44100
    # RECORD_SECONDS = 5
    # WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * length)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    write_audio(
        p, 
        frames, 
        n_channels=CHANNELS, 
        format=FORMAT, 
        sample_rate=RATE)

    return

def write_audio(p, frames: list, n_channels, format, sample_rate) -> None:

    wf = wave.open(audio_file, 'wb')
    wf.setnchannels(n_channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return

def speech_to_text() -> None:

    pipe = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")

    text_result = pipe(audio_file)

    print(text_result['text'])

    return


# Run script.
main(stop_time=10)
# record_sample(length=5)