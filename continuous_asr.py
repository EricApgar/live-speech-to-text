import argparse

from audio_object import Audio
from AsrModels.openAiWhisper import OpenAiWhisperModel
from AsrModels.facebook960hr import Facebook960hrModel
from AsrModels.fasterWhisper import FasterWhisperModel
from list_recording_devices import get_device_sample_rates


MODEL_OPTIONS = {
    'facebook_960hr': Facebook960hrModel(),
    'openai_whisper': OpenAiWhisperModel(),
    'faster_whisper': FasterWhisperModel()}


def get_model_by_name(model_name: str):
    '''
    Currently supported models.
    '''

    if model_name not in MODEL_OPTIONS:
        raise ValueError(f"Model '{model_name}' is not supported. Please specify a valid model name.")

    return MODEL_OPTIONS[model_name]


def get_recording_sample_rate(input_device_index: int, target_rate_hz: int) -> int:
    '''
    Get the sample rate for this recording device closest to a target rate.

    The ASR models currently all require input data at 16000 Hz, so if the
    current recording device can't sample at that rate, then sample at the
    rate closes to 16000 that it is capable of and resample to 16000.
    '''

    all_rates_hz = get_device_sample_rates(input_device_index=input_device_index)

    rate_hz = min(all_rates_hz, key=lambda rate: abs(rate - target_rate_hz))

    return rate_hz


def main(input_device_index: int=None, model_name: str="openai_whisper"):

    # Initialize the model and get its sample rate.
    model = get_model_by_name(model_name)
    model_rate_hz = model.get_sample_rate()

    # Get the recording device sample rate closest to the model's sample rate.
    record_rate_hz = get_recording_sample_rate(input_device_index=input_device_index, target_rate_hz=model_rate_hz)

    audio = Audio(input_device_index=input_device_index)
    
    print('Setting silence threshold... shhh...')
    audio.set_silence_threshold(rate_hz=record_rate_hz)
    print('Done.\n')

    print('Waiting to transcribe...\n')
    while True:
        
        audio.record_activity(rate_hz=record_rate_hz)
        audio.resample_audio(rate_hz=model_rate_hz)
        
        # Send audio data to model to be transcribed.
        text = model.transcribe_audio_array(audio_array=audio.data, sample_rate_hz=audio.rate_hz)

        print(text[0])

        if 'stop' in text[0].lower():
            break


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run live speech-to-text transcription.")
    
    parser.add_argument(
        "--input_device_index",
        type=int,
        default=None,
        help="Optional index of the audio input device to use (default: None).")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai_whisper",
        choices=list(MODEL_OPTIONS.keys()),
        help="Name of the ASR model to use (default: 'openai_whisper').")

    args = parser.parse_args()

    main(input_device_index=args.input_device_index, model_name=args.model_name)