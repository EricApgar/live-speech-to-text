import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
sys.path.append(repo_dir)

from audio_object import Audio
from AsrModels.openAiWhisper import OpenAiWhisperModel
from continuous_asr import get_recording_sample_rate


model = OpenAiWhisperModel()
model_rate_hz = model.get_sample_rate()
print(f'\n\nModel: OpenAI Whisper, Sample Rate (Hz): {model_rate_hz}\n\n')

# Get the recording device sample rate closest to the model's sample rate.
record_rate_hz = get_recording_sample_rate(input_device_index=11, target_rate_hz=model_rate_hz)
print(f'\n\nRecording Rate (Hz): {record_rate_hz}\n\n')

A = Audio(input_device_index=11)

frames_per_buffer_test = [256, 512, 1024]

for frames_per_buffer in frames_per_buffer_test:
    
    try:
        A._open_stream(rate_hz=record_rate_hz, frames_per_buffer=frames_per_buffer)
        data = A._read_stream()
        A._close_stream()

        print(f'\n\nValid frames_per_buffer: {frames_per_buffer}\n\n')
    except:
        pass

print('\n\nSetting silence threshold... shhh...')
A.set_silence_threshold(rate_hz=record_rate_hz)
print('Done.\n\n')

print('\n\nRecording Audio...')
A.record(rate_hz=record_rate_hz)
print('Done.\n\n')

print('\n\nResampling Audio...')
A.resample_audio(rate_hz=model_rate_hz)
print('Done.\n\n')

print('\n\nRecording Activity...')
A.record_activity(rate_hz=record_rate_hz)
print('Done.\n\n')