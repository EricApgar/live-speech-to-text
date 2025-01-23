import argparse

from audio_object import Audio
from AsrModels.openAiWhisper import OpenAiWhisperModel
from AsrModels.facebook960hr import Facebook960hrModel
from AsrModels.fasterWhisper import FasterWhisperModel
from list_recording_devices import get_sample_rates


def get_model_by_name(model_name):

    if model_name.lower() == "facebook_960hr":
        model = Facebook960hrModel()
    elif model_name.lower() == "openai_whisper":
        model = OpenAiWhisperModel()
    elif model_name.lower() == "faster_whisper":
        model = FasterWhisperModel()
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Please specify a valid model name.")

    return model


def is_rate_valid(rate_hz: int, input_device_index: str) -> bool:

    is_valid = rate_hz in get_sample_rates(input_device_index=input_device_index)

    return is_valid


def main(input_device_index: int=None, model_name: str="openai_whisper", rate_hz: int=16000):

    if not is_rate_valid(rate_hz=rate_hz, input_device_index=input_device_index):
        raise ValueError(f'Input device (index = {input_device_index}) cannot sample at given rate ({rate_hz})!')

    model = get_model_by_name(model_name)
    audio = Audio(input_device_index=input_device_index)
    
    print('Setting silence threshold... shhh...')
    audio.set_silence_threshold()
    print('Done.\n')

    print('Waiting to transcribe...\n')
    while True:
        
        audio.record_activity(rate_hz=rate_hz)
        
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
        help="Name of the ASR model to use (default: 'openai_whisper').")
    
    parser.add_argument(
        "--rate_hz",
        type=int,
        default=16000,
        help="Sampling rate for recording in Hertz (default: 16000).")

    args = parser.parse_args()

    main(
        input_device_index=args.input_device_index,
        model_name=args.model_name,
        rate_hz=args.rate_hz)