import time
from transformers import pipeline
import os
import pathlib
from record_audio import record_sample


repo_folder = os.path.dirname(str(pathlib.Path(__file__).absolute()))
audio_file = os.path.join(repo_folder, 'Temp Audio Files', 'audio.wav')

def main(stop_time: float=10) -> None:

    start_time = time.time()

    while (time.time() - start_time) < stop_time:

        record_sample(length=2)
        speech_to_text()

    return

def speech_to_text() -> None:

    pipe = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")

    text_result = pipe(audio_file)

    print(text_result['text'])

    return


# Run script.
main(stop_time=10)
