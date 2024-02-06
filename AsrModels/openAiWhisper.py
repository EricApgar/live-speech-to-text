from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from transformers import pipeline


class OpenAiWhisperModel():

    def __init__(self, size_ext: str='tiny.en'):
        '''
        size_ext: size of whisper model ('tiny, 'small', etc.) and
            extension if needed ('en'). Ex: 'tiny.en', 'small', etc.
        '''
        
        self.model = WhisperForConditionalGeneration.from_pretrained(f'openai/whisper-{size_ext}')
        self.processor = WhisperProcessor.from_pretrained(f'openai/whisper-{size_ext}')

    def transcribe_audio_array(self, audio_array: np.array, sample_rate_hz: int, max_new_tokens: int=20) -> list:
        '''
        Returns a list of strings (or really a single string in a list) of the
        transcribed audio signal's text.

        ---
        audio_array: numpy array of audio signal data.
        sample_rate_hz: sample rate of passed audio data.
        max_new_tokens: TODO.
        '''

        input_features = self.processor(audio_array, return_tensors="pt", sampling_rate=sample_rate_hz).input_features

        # Generate token ids.
        predicted_ids = self.model.generate(input_features, max_new_tokens=max_new_tokens)

        # Decode token ids to text.
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription
    
    @staticmethod
    def transcribe_audio_file(audio_file: str) -> list:
        '''
        Uses the transformers pipeline to transcribe audio files.

        audio_file: full file path to audio file.
        '''

        model = pipeline("automatic-speech-recognition", "openai/whisper-tiny")
        transcription = model(audio_file)  # Optional Param: chunk_length_s=30

        return transcription["text"]