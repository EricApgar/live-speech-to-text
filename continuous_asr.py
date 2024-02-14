import sys
import os
from HelperFunctions.system_info import is_raspberry_pi

# Turn off screen print ALSA warnings if running on RPi.
if is_raspberry_pi():

    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    # To restore stderr if necessary
    # sys.stderr = stderr

    print('YOU ARE ON RPi')

from audio_object import Audio
from AsrModels.openAiWhisper import OpenAiWhisperModel


audio = Audio()
model = OpenAiWhisperModel()

print('Setting silence threshold... shhh...')
audio.set_silence_threshold()
print('Done.\n')

print('Waiting to transcribe...\n')
while True:
    
    audio.record_activity()
    
    text = model.transcribe_audio_array(audio_array=audio.data, sample_rate_hz=audio.rate_hz)

    print(text[0])

    if 'stop' in text[0].lower():
        break