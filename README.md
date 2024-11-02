# live-speech-to-text
Speech to text transcription. Ambient sound is recorded and streamed to an Automatic Speech Recognition (ASR) model that will transcribe the audio to text. 

## Highlights
* Works completely offline and runs locally.
   * No API's or other services needed.
* Capable of transcribing both pre-recorded audio files (i.e. a .flac file) and running a live stream of audio to perform real time ASR.
* Multiple Hugging Face ASR model options available.

# Requirements

## Microphone Setup
On Windows, there was no additional setup required beyond connecting a recording device. On Linux (Raspberry pi, etc.), there's some additional setup required to connect and specify the recording HAT. For details on setting up recording devices on the RPi, see [this wiki](https://github.com/EricApgar/raspberry-pi-how-to/wiki/Audio-Recording-Basics).

## Software

### Python 3.11.1
Other versions of Python may work but not guranteed.

Create a virtual environment and install from "requirements.txt".
```
pip install -r requirements.txt
```

## OS

### Supported
* Linux (Raspberry Pi OS Bookworm)
   * Performance on other Linux distributions has not been tested.
* Windows 11 (see note below about minor differences)
 
### Windows Differences
There are two packages that need to be installed separately for *full* functionality on Windows, however most of the tools will work without them. These libraries haven't been fully tested with this repo and may or may not work after installing.
* cudnn_ops_infer64_8.dll
  * Required to run only the "Systran/faster-whisper-tiny.en" model.
* ffmpeg
  * This only affects transcription of a recorded audio file (i.e. .flac, .wav). It doesn't affect the continuous ASR ability which is done by streaming raw audio data. 

## Hardware
The table below are combinations of Hardware and Models that have been tested. These tests were done using "continuous_asr.py" to determine the working ability of the model. A model that was too large and caused freezing or other problems has no time listed for Transcription Time.

See [Hugging Face](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads) for details on the models.

### Successfully tested hardware/model combinations
| Hardware | OS | GPU | Model | Transcription Time (single spoken word) |
|-|-|-|-|-|
| Raspberry Pi 4B (2 GB) | Raspberry Pi OS *Bookworm* | - | facebook/wav2vec2-base-960hr | - |
| Raspberry Pi 4B (2 GB) | Raspberry Pi OS *Bookworm* | - | openai/whisper-tiny.en | ~4 sec |
| Raspberry Pi 4B (2 GB) | Raspberry Pi OS *Bookworm* | - | Systran/faster-whisper-tiny.en | ~3 sec |
| Desktop PC | Windows 10 | 3060 Ti | facebook/wav2vec2-base-960hr | ~.15 sec |
| Desktop PC | Windows 10 | 3060 Ti | openai/whisper-tiny.en | ~.55 sec |

### Audio Recording Hardware
All performance benchmarks were done using the recording devices below.

| Hardware | Recording Device |
|-|-|
| Raspberry Pi 4B (2 GB) | Seeed Studio 2-mic HAT |
| Desktop PC | Bluetooth headphones with built in mic |

# How to Run
## continuous_asr.py
Continually streams live captured audio to the model and transcribes real time.

```
python continuous_asr.py
```

## tests/test_all.py
A decent run through of the capabilities of a couple different models and the main audio manipulation class.
```
python test_all.py
```

# Notes


# Technical Details

## How to stream sound samples to model

* Record super short samples of sound (~.1 seconds in length).
* Analyze each sample for exceeding a sound threshold above an adjustable limit.
* Continuously build a buffer of sound samples that contain sound above the threshold.
* Once you have a sample where the threshold is NOT exceeded, assume that this is the end of the phrase or word and stop building the buffer.
* Send the whole buffer to the model directly to be transcribed into text.
* Go back to listening for sound.
