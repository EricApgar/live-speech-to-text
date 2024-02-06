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

    if 'stop' in text[0].lower():
        break