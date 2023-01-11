from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import pyaudio
import time
import numpy as np

# For Debugging.
from scipy.io.wavfile import write as wav_write
import pathlib
import os
import matplotlib.pyplot as plt


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

def detect_and_sample(show_plot: bool=False) -> np.array:

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

    ready_for_model = False  # Initialize to no good sample found.
    full_sample = []

    while True:  # Break out after creating a good sample.
        
        sample = read_audio_stream(stream=stream, chunk_size=CHUNK, read_time_s=.2)

        is_active, ends_dead = activity_levels(sample=sample)

        if is_active:  # Has something that looks like non-background sound.
            full_sample.extend(sample)  # Add snippet to the full sample array.
            ready_for_model = True

        if ready_for_model and ends_dead:  # Means dead zone found after a sample with activity.
            break

    # Cleanup stream.
    stream.stop_stream()
    stream.close()
    p.terminate()

    if show_plot:  # DEBUGGING:

        # Write sample to wav.
        repo_folder = os.path.dirname(str(pathlib.Path(__file__).absolute()))
        audio_file = os.path.join(repo_folder, 'Temp Audio Files', 'audio.wav')
        wav_write(audio_file, RATE, np.array(full_sample).astype(np.int16))
        
        # Plotting sample.
        plt.plot(full_sample)
        plt.show()

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
    Determine if an audio sample has any activity, and if it ends in a dead zone. 
    If a sample has activity
    '''

    SOUND_THRESHOLD = 1000  # Anything below this is considered background noise. Above is considered "active".
    PERCENT_ACTIVE_REQUIRED = 10  # This percent samples over the SOUND_THRESHOLD is considered an "active" clip. 
    PERCENT_LOOK_BACK = 25  # If there no above-noise-threshold elements in this "back percent" of the sample, its the end of the clip.

    is_active = False  # Initial state for sample has active points.
    ends_dead = True  # Initial for the clip ending in a non-active state.

    sample_length = len(sample)

    n_look_back = int(np.round(PERCENT_LOOK_BACK/100 * sample_length))  # Number of frames to look back on.
    num_active_required = int(np.round(PERCENT_ACTIVE_REQUIRED/100 * sample_length))  # Num active frames required.

    # This is a list of the all the indexes of the frames that were considered "active".
    i_active = [i for i, value in enumerate(sample) if abs(value) > SOUND_THRESHOLD]

    if len(i_active) >= num_active_required:  # Total number of active frames is above threshold...
        is_active = True
    
        # If the last active frame is later than the percent look back frame...
        if i_active[-1] > (sample_length - n_look_back):  # Last active value.
            ends_dead = False

    return is_active, ends_dead

def transcribe_audio(audio_array, model, processor) -> list:

    SAMPLING_RATE = 16000 # Standard, but should be same as in detect_and_sample().

    # Tokenize.
    input_values = processor(audio_array, return_tensors="pt", padding="longest", sampling_rate=SAMPLING_RATE).input_values  # Batch size 1.

    # Retrieve logits.
    logits = model(input_values).logits

    # Take argmax and decode.
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription

def main(run_time: float=10, show_plot: bool=False) -> None:

    # Load model and tokenizer.
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    start_time = time.time()

    while (time.time() - start_time) < run_time:

        print('Waiting for sound...')

        data = detect_and_sample(show_plot=show_plot)

        transcription = transcribe_audio(data, model=model, processor=processor)

        print(transcription[0])

    return


# MAIN:
main(run_time=100, show_plot=False)


