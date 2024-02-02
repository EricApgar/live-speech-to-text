import pyaudio

def get_sample_width(pyaudio_format) -> int:
    '''
    2: This is the width (in bytes) of the audio samples. 
    In this case, 2 indicates that the audio data is in 
    16-bit format (as 16 bits equals 2 bytes). The width 
    depends on how the audio is configured (specifically 
    the FORMAT in PyAudio, which is set to pyaudio.paInt16 
    for 16-bit audio).
    '''

    format_to_width = {
        pyaudio.paFloat32: 4,
        pyaudio.paInt32: 4,
        pyaudio.paInt24: 3,
        pyaudio.paInt16: 2,
        pyaudio.paInt8: 1,
        pyaudio.paUInt8: 1
    }

    result = format_to_width.get(pyaudio_format, None)

    return result  # Returns None if format is not found.
