import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
sys.path.append(repo_dir)

from audio_object import Audio
from AsrModels.facebook960hr import Facebook960hrModel
from AsrModels.openAiWhisper import OpenAiWhisperModel

import time

A = Audio()

print('Setting noise level...')
A.set_noise_level()
print('Done.')

print('Recording sample...')
A.record()
print('Done.')

print('Recording active sample...')
A.record_activity()
print('Done.')

print('Saving active sample...')
A.save_as('audio.flac')
print('Done.')

print('Loading Facebook model...')
model = Facebook960hrModel()
print('Done.')

print('Transcribing audio array...')
start_time = time.time()
text = model.transcribe_audio(A.data, A.rate_hz)
print(f'Trasncription: {text}')
print(f'Done in {time.time() - start_time:.2f} sec.')

print('Transcribing audio file...')
start_time = time.time()
text = model.transcribe_audio_file('audio.flac')
print(f'Trasncription: {text}')
print(f'Done in {time.time() - start_time:.2f} sec.')

print('Loading OpenAi Whisper model...')
model = OpenAiWhisperModel()
print('Done.')

print('Transcribing audio array...')
start_time = time.time()
text = model.transcribe_audio(A.data, A.rate_hz)
print(f'Trasncription: {text}')
print(f'Done in {time.time() - start_time:.2f} sec.')

print('Transcribing audio file...')
start_time = time.time()
text = model.transcribe_audio_file('audio.flac')
print(f'Trasncription: {text}')
print(f'Done in {time.time() - start_time:.2f} sec.')