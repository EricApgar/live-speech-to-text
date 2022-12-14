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

    start_time = time.time()  # REMOVE.
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)
    elapsed_time = time.time() - start_time  # REMOVE.
    print(f'Time to open stream: {elapsed_time:2f}')  # REMOVE.

    numpy_data = []

    start_time = time.time()  # REMOVE.
    while (time.time() - start_time) < length:
    # for _ in range(0, int(RATE / CHUNK * length)):
        data = stream.read(CHUNK)
        numpy_data.extend(np.frombuffer(data, dtype=np.int16))
    elapsed_time = time.time() - start_time  # REMOVE.
    print(f'Time to build sound array: {elapsed_time:2f}')  # REMOVE.

    start_time = time.time()  # REMOVE.
    stream.stop_stream()
    stream.close()
    p.terminate()
    elapsed_time = time.time() - start_time  # REMOVE.
    print(f'Time to cleanup stream: {elapsed_time:2f}')  # REMOVE.

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

    full_transcription = []

    while (time.time() - start_time) < stop_time:

        print_time = time.time()
        data = record_sample(length=2)
        elapsed_time = time.time() - print_time
        print(f'Time to record: {elapsed_time:2f}')

        print_time = time.time()
        transcription = transcribe_audio(data, model=model, processor=processor)
        elapsed_time = time.time() - print_time
        print(f'Time to Transcribe: {elapsed_time:2f}')

        full_transcription.extend(transcription)

    print(' '.join(full_transcription))

    return


# MAIN:
main()

# working_test()

