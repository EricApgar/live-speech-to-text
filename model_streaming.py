from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import pyaudio
import time
import numpy as np


def working_test():

    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        
    # load dummy dataset and read soundfiles
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

    audio_array = ds[0]["audio"]["array"]

    # tokenize
    input_values = processor(audio_array, return_tensors="pt", padding="longest").input_values  # Batch size 1

    # retrieve logits
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    print(transcription)

    return

def record_sample(length: float=1) -> None:

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  #2
    RATE = 16000  #44100

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)

    numpy_data = []

    for _ in range(0, int(RATE / CHUNK * length)):
        data = stream.read(CHUNK)
        numpy_data.extend(np.frombuffer(data, dtype=np.int16))

    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.double(numpy_data)

def transcribe_audio(audio_array, model, processor):

    # tokenize
    input_values = processor(audio_array, return_tensors="pt", padding="longest").input_values  # Batch size 1

    # retrieve logits
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription

def main(stop_time: float=10) -> None:

    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    start_time = time.time()

    while (time.time() - start_time) < stop_time:

        data = record_sample(length=2)

        transcription = transcribe_audio(data, model=model, processor=processor)

        print(transcription)

    return


# MAIN:
main()

# working_test()

