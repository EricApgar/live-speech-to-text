from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
import torch
from transformers import pipeline


class Facebook960hrModel():

    def __init__(self):
        
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def transcribe_audio_array(self, audio_array: np.array, sample_rate_hz: int) -> list:
        '''
        Returns a list of strings (or really a single string in a list) of the
        transcribed audio signal's text.

        audio_obj: instance of the audio class. This must have sound 
        max_new_tokens: TODO
        '''
        
        # SAMPLING_RATE = 16000 # Standard, but should be same as in detect_and_sample().

        # Tokenize.
        input_values = self.processor(
            audio_array,
            return_tensors="pt",
            padding="longest",
            sampling_rate=sample_rate_hz).input_values  # Batch size 1.
        
        # Just the processor command returns a dictionary with a single key: "input_values" which is a tensor with shape [1, 12288]
        # or really [1, the length of the audio_array]. The "padding" parameter doesn't seem to do anything.

        # print(f'Input Values Shape: {input_values.shape}')

        # Retrieve logits.
        logits = self.model(input_values).logits

        # Take argmax and decode.
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)

        return transcription
    
    @staticmethod
    def transcribe_audio_file(audio_file: str):
        '''
        Uses the transformers pipeline to transcribe audio files.

        audio_file: full file path to audio file.
        '''

        model = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
        transcription = model(audio_file)  # , chunk_length_s=30
        # print(transcription["text"][:500])
        # print(transcription["text"])

        return transcription["text"]