import os
import pathlib
import pyaudio
import wave


def record_sample(length: float=1, save_path: str=None) -> None:

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  #2
    RATE = 16000  #44100

    if save_path is None:
        repo_folder = os.path.dirname(str(pathlib.Path(__file__).absolute()))
        save_path = os.path.join(repo_folder, 'Temp Audio Files', 'audio.wav')

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []
    for _ in range(0, int(RATE / CHUNK * length)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Done.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    write_audio(
        p=p,
        frames=frames,
        save_path=save_path,
        n_channels=CHANNELS, 
        format=FORMAT, 
        sample_rate=RATE)

    return

def write_audio(
        p: pyaudio.PyAudio,
        frames: list,
        save_path: str,
        n_channels: int=1,
        format: int=pyaudio.paInt16,
        sample_rate: int=16000) -> None:

    wf = wave.open(save_path, 'wb')
    wf.setnchannels(n_channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return


if __name__ == '__main__':
    record_sample(length=5)