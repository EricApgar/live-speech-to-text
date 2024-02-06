import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
sys.path.append(repo_dir)

from audio_object import Audio
from AsrModels.facebook960hr import Facebook960hrModel
from AsrModels.openAiWhisper import OpenAiWhisperModel

import time
import platform


'''
This will test the basic functionality of the Audio class, as well as the
basic ASR capabilities of the current models.
'''

def build_audio() -> Audio:
    
    A = Audio()

    return A

def set_silence(A: Audio):

    print('Setting silence threshold...')
    A.set_silence_threshold()
    print('Done.\n')

    return

def record_sample(A: Audio):
    
    print('Recording audio sample...')
    A.record()
    print('Done.\n')

    return

def record_active_sample(A: Audio):

    print('Recording active sample...')
    A.record_activity(max_sample_length_s=3)
    print('Done.\n')

    return

def plot_sample(A: Audio):
        
    print('Plotting active sample...')
    A.plot()
    print('Done.\n')

    return

def save_sample(A: Audio):

    print('Saving active sample...')
    A.save_as('audio.flac')
    print('Done.\n')

    return

def load_model_facebook() -> Facebook960hrModel:

    print('Loading Facebook model...')
    model = Facebook960hrModel()
    print('Done.\n')

    return model

def transcribe_sample(A: Audio, model):
        
    print('Transcribing audio array...')
    start_time = time.time()
    text = model.transcribe_audio_array(A.data, A.rate_hz)
    print(f'Trasncription: {text}')
    print(f'Done in {time.time() - start_time:.2f} sec.\n')

    return

def transcribe_flac(model):

    # NOTE: Transcribing audio files using transformers pipeline requires ffmpeg to be installed.
    # This is easy on linux, annoying (I haven't bothered) on Windows.
    if not platform.system() == 'Windows':
        print('Transcribing audio file...')
        start_time = time.time()
        text = model.transcribe_audio_file(audio_file='audio.flac')
        print(f'Trasncription: {text}')
        print(f'Done in {time.time() - start_time:.2f} sec.\n')

    return

def load_model_whisper() -> OpenAiWhisperModel:

    print('Loading OpenAi Whisper model...')
    model = OpenAiWhisperModel()
    print('Done.\n')

    return model

def test_all():

    A = build_audio()

    set_silence(A)
    record_sample(A)
    record_active_sample(A)
    plot_sample(A)
    save_sample(A)

    model = load_model_facebook()
    transcribe_sample(A, model)
    transcribe_flac(model)

    model = load_model_whisper()
    transcribe_sample(A, model)
    transcribe_flac(model)

    return


if __name__ == "__main__":
    test_all()