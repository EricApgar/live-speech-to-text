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
            print(f"\tSample Rates: {get_sample_rates(input_device_index=i)}\n")

    p.terminate()

    return


def get_sample_rates(input_device_index: int) -> list:

    COMMON_RATES = [
        8000,
        16000,
        22050,
        32000,
        44100,
        48000,
        96000,
        192000]
    
    p = pyaudio.PyAudio()

    sample_rates = []
    
    for rate in COMMON_RATES:
        try:
            stream = p.open(
                rate=rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                input_device_index=input_device_index)
            
            stream.close()

            sample_rates.append(rate)

        except Exception:
            continue

    p.terminate()

    return sample_rates


if __name__ == '__main__':
    list_recording_devices()