import pyaudio


def list_recording_devices():

    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        num_channels = device_info['maxInputChannels']
        
        if num_channels != 0:  # Only list recording capable devices.
            print(f"Device Index: {i}")
            print(f"Device Name: {device_info['name']}")
            print(f"Max Input Channels: {device_info['maxInputChannels']}\n")

    p.terminate()


if __name__ == '__main__':
    list_recording_devices()