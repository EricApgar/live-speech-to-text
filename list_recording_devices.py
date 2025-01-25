import pyaudio


def list_recording_devices():

    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        num_channels = device_info['maxInputChannels']
        
        if num_channels != 0:  # Only list recording capable devices.
            print(f"Device Index: {i}")
            print(f"\tDevice Name: {device_info['name']}")
            print(f"\tMax Input Channels: {device_info['maxInputChannels']}")
            print(f"\tSample Rates: {get_device_sample_rates(input_device_index=i)}\n")

    p.terminate()

    return


def get_device_sample_rates(input_device_index: int) -> list:

    COMMON_RATES = [
        8000,
        16000,
        22050,
        32000,
        44100,
        48000,
        96000]
    
    p = pyaudio.PyAudio()

    sample_rates = []
    
    for rate_hz in COMMON_RATES:
        try:
            # stream = p.open(
            #     rate=rate_hz,
            #     channels=1,
            #     format=pyaudio.paFloat32,
            #     input=True,
            #     input_device_index=input_device_index)
            
            stream = p.open(
                input=True,  # Tells the stream you are opening to record data.
                rate=rate_hz,  # Sample Rate.
                channels=1,  # If not set, this should default to the recording device's default.
                frames_per_buffer=1024,
                format=pyaudio.paFloat32,
                input_device_index=input_device_index,
                start=True)
            
            stream.close()

            sample_rates.append(rate_hz)

        except Exception:
            continue

    p.terminate()

    return sample_rates


if __name__ == '__main__':
    list_recording_devices()