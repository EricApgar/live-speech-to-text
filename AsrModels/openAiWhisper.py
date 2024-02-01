from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np


class OpenAiWhisperModel():

    def __init__(self):
        
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

    def transcribe_audio(self, audio_array: np.array, sample_rate_hz: int, max_new_tokens: int=20) -> list:
        '''
        Returns a list of strings (or really a single string in a list) of the
        transcribed audio signal's text.

        audio_obj: instance of the audio class. This must have sound 
        max_new_tokens: TODO
        '''

        # SAMPLING_RATE = 16000 # Standard, but should be same as in detect_and_sample().

        # input_features = processor(audio_array, return_tensors="pt", sampling_rate=SAMPLING_RATE).input_features  # Batch size 1.
        input_features = self.processor(audio_array, return_tensors="pt", sampling_rate=sample_rate_hz).input_features  # Batch size 1.
        # Returns a dictionary with a single key: "input_features" which is a tensor with shape [1, 80, 3000]

        # generate token ids
        predicted_ids = self.model.generate(input_features, max_new_tokens=max_new_tokens)
        # logits = model(input_features).logits
        # predicted_ids2 = torch.argmax(logits, dim=-1)

        # decode token ids to text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription