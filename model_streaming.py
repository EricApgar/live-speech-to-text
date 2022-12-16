from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import pyaudio
import time
import numpy as np

# import wave
# import pathlib
# import os


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

def detect_and_sample() -> np.array:

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  #2
    RATE = 16000  #44100

    # start_time = time.time()  # REMOVE.
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)

    ready_for_model = False  # Initialize to no good sample found.
    full_sample = []

    while True:
        
        sample = read_audio_stream(stream=stream, chunk_size=CHUNK, read_time_s=.2)

        is_active, ends_dead = activity_levels(sample=sample)

        if is_active:
            full_sample.extend(sample)
            ready_for_model = True

        if ready_for_model and ends_dead:  # Means dead zone found after a sample with activity.
            break

    # Cleanup stream.
    stream.stop_stream()
    stream.close()
    p.terminate()

    # write_audio(
    #     p=p, 
    #     frames=full_raw, 
    #     n_channels=CHANNELS, 
    #     format=FORMAT, 
    #     sample_rate=RATE)

    return full_sample

def read_audio_stream(stream, read_time_s: float=.1, chunk_size: int=1024) -> np.array:

    full_data = []

    start_time = time.time()

    while (time.time() - start_time) < read_time_s:

        raw_sample = stream.read(chunk_size)  # Time to read CHUNK is approx (1/SAMPLE_RATE * CHUNK) seconds.
        num_sample = np.double(np.frombuffer(raw_sample, dtype=np.int16))
        full_data.extend(num_sample)

    return full_data

def activity_levels(sample: np.array):
    '''
    Determine if a sample has any activity, and if it ends in a dead zone. 
    This means that it's the end of a word. 
    '''

    SOUND_THRESHOLD = 1000
    PERCENT_ACTIVE_REQUIRED = 10
    PERCENT_LOOK_BACK = 25

    is_active = False  # Initial values.
    ends_dead = True  # Initial values.

    sample_length = len(sample)

    n_look_back = int(np.round(PERCENT_LOOK_BACK/100 * sample_length))
    num_active_required = int(np.round(PERCENT_ACTIVE_REQUIRED/100 * sample_length))

    i_active = [i for i, value in enumerate(sample) if abs(value) > SOUND_THRESHOLD]

    if len(i_active) >= num_active_required:
        is_active = True
    
        if i_active[-1] > (sample_length - n_look_back):  # Last active value.
            ends_dead = False

    return is_active, ends_dead

def transcribe_audio(audio_array, model, processor) -> list:

    SAMPLING_RATE = 16000 # Standard, but should be same as in detect_and_sample().

    # Tokenize.
    input_values = processor(audio_array, return_tensors="pt", padding="longest", sampling_rate=16000).input_values  # Batch size 1.

    # Retrieve logits.
    logits = model(input_values).logits

    # Take argmax and decode.
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription

def main(run_time: float=10) -> None:

    # Load model and tokenizer.
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    start_time = time.time()

    while (time.time() - start_time) < run_time:

        print('Waiting for sound...')

        data = detect_and_sample()

        transcription = transcribe_audio(data, model=model, processor=processor)

        print(transcription[0])

    return

def write_audio(p, frames: list, n_channels, format, sample_rate) -> None:

    repo_folder = os.path.dirname(str(pathlib.Path(__file__).absolute()))
    audio_file = os.path.join(repo_folder, 'Temp Audio Files', 'audio.wav')

    wf = wave.open(audio_file, 'wb')
    wf.setnchannels(n_channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return


# MAIN:
main(run_time=100)


