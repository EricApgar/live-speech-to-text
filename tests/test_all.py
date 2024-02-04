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


A = Audio()

print('Setting noise level...')
A.set_silence_threshold()
print('Done.\n')

print('Recording sample...')
A.record()
print('Done.\n')

print('Plotting sample...')
A.plot()
print('Done.\n')

print('Recording active sample...')
A.record_activity(max_sample_length_s=3)
print('Done.\n')

print('Saving active sample...')
A.save_as('audio.flac')
print('Done.\n')

print('Loading Facebook model...')
model = Facebook960hrModel()
print('Done.\n')

print('Transcribing audio array...')
start_time = time.time()
text = model.transcribe_audio_array(A.data, A.rate_hz)
print(f'Trasncription: {text}')
print(f'Done in {time.time() - start_time:.2f} sec.\n')

# NOTE: Transcribing audio files using transformers pipeline requires ffmpeg to be installed.
# This is easy on linux, annoying (I haven't bothered) on Windows.
if not platform.system() == 'Windows':
    print('Transcribing audio file...')
    start_time = time.time()
    text = model.transcribe_audio_file(audio_file='audio.flac')
    print(f'Trasncription: {text}')
    print(f'Done in {time.time() - start_time:.2f} sec.\n')

print('Loading OpenAi Whisper model...')
model = OpenAiWhisperModel()
print('Done.\n')

print('Transcribing audio array...')
start_time = time.time()
text = model.transcribe_audio_array(A.data, A.rate_hz)
print(f'Trasncription: {text}')
print(f'Done in {time.time() - start_time:.2f} sec.\n')

# NOTE: Transcribing audio files using transformers pipeline requires ffmpeg to be installed.
# This is easy on linux, annoying (I haven't bothered) on Windows.
if not platform.system() == 'Windows':
    print('Transcribing audio file...')
    start_time = time.time()
    text = model.transcribe_audio_file(audio_file='audio.flac')
    print(f'Trasncription: {text}')
    print(f'Done in {time.time() - start_time:.2f} sec.\n')