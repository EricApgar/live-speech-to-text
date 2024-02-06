# LiveSpeechToText
Speech to text transcription. Ambient sound is recorded and streamed to an Automatic Speech Recognition (ASR) model that will transcribe the audio to text. 

## Highlights:
* Works completely offline and runs locally.
   * No API's or other services needed.
* Capable of transcribing both pre-recorded audio files (i.e. a .flac file) and running a live stream of audio to perform real time ASR.
* Multiple Hugging Face ASR model options available.

# Requirements:

## Software:

### Python 3.11.1
Other versions of Python may work but not guranteed.

Create a virtual environment and install from "requirements.txt".
```
pip install -r requirements.txt
```

See **Notes** section below for additional Linux (RPi) installation requirements. 

## OS:

### Supported:
* Windows 11
* Linux (Raspberry Pi OS Bookworm)
   * For details on setting up recording devices on the RPi, see [this wiki](https://github.com/EricApgar/HowToRaspberryPi/wiki).

## Hardware:
The table below are combinations of Hardware and Models that have been tested. These tests were done using "continuous_asr.py" to determine the working ability of the model. A model that was too large and caused freezing or other problems has no time listed for Transcription Time.

See [Hugging Face](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads) for details on the models.

### Successfully tested hardware/model combinations:
| Hardware | OS | GPU | Model | Transcription Time |
|-|-|-|-|-|
| Raspberry Pi 4B (2 GB) | Raspberry Pi OS *Bookworm* | - | facebook/wav2vec2-base-960hr | - |
| Raspberry Pi 4B (2 GB) | Raspberry Pi OS *Bookworm* | - | openai/whisper-tiny.en | ~3 sec |
| Desktop PC | Windows 11 | 3060 Ti | facebook/wav2vec2-base-960hr | ~.15 sec |
| Desktop PC | Windows 11 | 3060 Ti | openai/whisper-tiny.en | ~.55 sec |

### Audio Recording Hardware:
| Hardware | Recording Device |
|-|-|
| Raspberry Pi 4B (2 GB) | Seeed Studio 2-mic HAT |
| Desktop PC | Bluetooth headphones with build in mic |

# How to Run:

## continuous_asr.py
Continually streams live captured audio to the model and transcribes real time.

```
(.venv) python continuous_asr.py
```

## tests/test_all.py
A decent run through of the capabilities of a couple different models and the main audio manipulation class.
```
(.venv) python test_all.py
```

# Notes:

## Linux:
On linux (RPi), There might be additional libraries to install on Linux.

You will likely have to run:
```
sudo apt install portaudio19-dev python3-pyaudio
```

This should fix an error when installing from requirements.txt that yells about portaudio installing when trying to install "PyAudio" which is a library in requirements.txt that is needed.

I believe you will also have to install ffmpeg. This will let you load and transcribe a .flac file (a capability that isn't currently supported on Windows because installing ffmpeg was annoying on Windows).
```
sudo apt install ffmpeg
```

### Setup Recording Device:
Select the speaker HAT by running:
```
alsamixer
```
and choosing the right soundcard for recording device using the F keys and menu options.

# Technical Details:

## How to stream sound samples to model:

* Record super short samples of sound (~.1 seconds in length).
* Analyze each sample for exceeding a sound threshold above an adjustable limit.
* Continuously build a buffer of sound samples that contain sound above the threshold.
* Once you have a sample where the threshold is NOT exceeded, assume that this is the end of the phrase or word and stop building the buffer.
* Send the whole buffer to the model directly to be transcribed into text.
* Go back to listening for sound.
