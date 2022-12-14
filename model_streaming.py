from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import pyaudio
import time
import numpy as np

# import matplotlib.pyplot as plt
# plt.plot([data])
# plot.show()


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

    # start_time = time.time()  # REMOVE.
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)
    # elapsed_time = time.time() - start_time  # REMOVE.
    # print(f'Time to open stream: {elapsed_time:2f}')  # REMOVE.

    ready_for_model = False
    full_sample = []

    start_time = time.time()  # REMOVE.
    while (time.time() - start_time) < length:
        
        sample = read_audio_stream(stream=stream, chunk_size=CHUNK, read_time_s=.1)
        
        if activity_has_stopped(sample=sample):

            full_sample.extend(sample)
            ready_for_model = True

        elif ready_for_model:  # Means dead zone found after a sample with activity.

            break


    numpy_data = []

    start_time = time.time()  # REMOVE.
    while (time.time() - start_time) < length:
    # for _ in range(0, int(RATE / CHUNK * length)):
        chunk_start = time.time()
        
        data = stream.read(CHUNK)  # Time to read CHUNK is approx (1/SAMPLE_RATE * CHUNK) seconds.
        audio_sample = np.double(np.frombuffer(data, dtype=np.int16))  # Numerical audio sample.
        numpy_data.extend(audio_sample)

        print(f'{time.time() - chunk_start:.3f} seconds to read chunk.')

        # Determine if the data is at the end of a word.
        threshold_10_percent = .1 * max(audio_sample)
        num_below = sum(abs(i) < threshold_10_percent for i in audio_sample[-10:])
        if num_below > 5:
            break

    # Continue to record samples and throw them away unless it looks like there's activity happening.
    # (Same as above statement) Dont pass on a sample to the model unless it looks like there's activity in it.
    # Once there is a sample that has activity, continue to add to it until there is a lull in activity.
    # Then stop recording and send that sample to the model.
    # Difficulties lie in fast detection of activity.

    # elapsed_time = time.time() - start_time  # REMOVE.
    # print(f'Time to build sound array: {elapsed_time:2f}')  # REMOVE.

    # start_time = time.time()  # REMOVE.
    stream.stop_stream()
    stream.close()
    p.terminate()
    # elapsed_time = time.time() - start_time  # REMOVE.
    # print(f'Time to cleanup stream: {elapsed_time:2f}')  # REMOVE.

    return numpy_data  #np.double(numpy_data)

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
        
        sample = read_audio_stream(stream=stream, chunk_size=CHUNK, read_time_s=.1)
        
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
    TODO: what happens if the dead zone is in the middle of the array? Hopefully
    not an issue. Looking for a longish pause signifying the end of a single word.
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

    # full_transcription = []

    while (time.time() - start_time) < stop_time:

        # print_time = time.time()
        # data = record_sample(length=2)
        # elapsed_time = time.time() - print_time
        # print(f'Time to record: {elapsed_time:2f}')

        data = detect_and_sample()

        # print_time = time.time()
        transcription = transcribe_audio(data, model=model, processor=processor)
        # elapsed_time = time.time() - print_time
        # print(f'Time to Transcribe: {elapsed_time:2f}')

        # full_transcription.extend(transcription)

        print(transcription[0])

    # print(' '.join(full_transcription))

    return


# MAIN:
main()

# working_test()

