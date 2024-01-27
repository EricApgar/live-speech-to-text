import numpy as np


class AsrModel:

    def __init__(self, name: str):
        
        self.model = None
        self.processor = None

    def transcribe_audio(self, audio_type: str):

        raise NotImplementedError("Subclass must implement abstract method")

        VALID_AUDIO_TYPES = {
            'array': np.array,  # Raw sound numpy array.
            'wav': str}  # File path to .wav.

        if audio_type not in VALID_AUDIO_TYPES:
            raise ValueError('Invalid audio_type!')

        return
    
class Facebook960hrModel(AsrModel):

    def transcribe_audio(self, audio_type: str):
        return super().transcribe_audio(audio_type)
    
class OpenAiWhisperModel(AsrModel):

    def transcribe_audio(self, audio_type: str):
        return super().transcribe_audio(audio_type)