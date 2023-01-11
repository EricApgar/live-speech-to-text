# LiveSpeechToText
Live speech to text transcription. Ambient sound is analyzed and parsed into sections of words that are streamed to a voice model that prints to the screen the text for the detected speech. Works completely offline - no streaming to a service or external connections to websites or 3rd party services needed.

There is a rudimentary (but adjustable) system for determining whether or not to send sound bytes to the text prediction model. If the sound level is over a certain (adjustable) threshold then a clip is passed to the model. The individual clip stops recording and gets passed to the model after it detects a lull in sound (adjustable time period) signifying the end of a word or sentence. 

# Requirements:

## Requires Python 3.7.8 to work. 

### See [here](https://www.python.org/downloads/release/python-378/) for python package download.

Virtual environment can be created based off of "requirements.txt".

# Methods:

## 1. Stream sound samples to model (RECOMMENDED)

### **Associated file:** model_streaming.py

Summary: This is by far the best method. 

* Record super short samples of sound (~.1 seconds in length).
* Analyze each sample for exceeding a sound threshold above an adjustable limit.
* Continuously build a buffer of sound samples that contain sound above the threshold.
* Once you have a sample where the threshold is NOT exceeded, assume that this is the end of the phrase or word and stop building the buffer.
* Send the whole buffer to the model directly to be transcribed into text.
* Go back to listening for sound.

## 2. Record a series of .wav clips to pass to the model.

### **Associated file:** cycle_wav.py

Summary: This samples super short clips of sound (n-seconds) and connects them together to create a single coherent audio sample. That single sample is then fed to the model.

* Super janky method.  
* Start recording n-second samples.  
* After every n seconds, save the audio as a .wav file.  
* Send that audio to the model to translate to text.  
* Delete the audio file.  
* Repeat.  


# Notes:
**The preferred method is to create a python virtual environment using "venv" and the "requirements.txt" file.**

If you make an environment through Anaconda, you could get some other issues regarding a need for the "ffmpeg" library. When you run the first time you might get an error about "ffmpeg". This library is needed by the transformers library, but doesn't technically show up as a referenced library with the typical import.

I didn't get this error creating a venv so maybe just don't use Anaconda.

```
conda install ffmpeg
```

# Future:
1. Detecting if a given clip had spoken voice in it as opposed to just a sound above the threshold.

# Problem Notes:

My stream of conscious notes on how to solve the problem.

* There's always a problem of cutting off a speaker mid-word if you sample audio at a set interval.
* You usually sample audio for x seconds (I've been doing x=1) and then pipe that sound to the model.
    * The problem is that the model takes about 1/5 of a second to run - longer for a longer segment. This delay isn't that long but by the time you predict and go back to sampling audio, you've lost a decent amount of time which could be in the middle of a word and that throws off all the predictions.
* If you wait to send audio to the model before you 
* Even if you don't sample at a set interval and you wait to detect a sound befor your ,

## Idea:
* Record audio for a very short amount of time (.2 seconds?) and immediately analyze the numpy array to see if the audio level has dropped to zero (or just a low number). This signals the end of a word. Make sure that the pause is long enough that its not just a pause in the middle of the word. If it's not the end, then keep reading audio samples into the array until you hit the dead zone.
    * Checking if the end of the array is a dead zone should be such a quick operation that the time away from recording audio should hopefully not cut out enough audio to mess with the sample. Have to check this though.
* Once the dead zone is confirmed, send the sample to the model.
* Repeat the process.
* This could be great for single words, but slurring words together or speaking quickly could be an issue. Getting the timing right between the words could be tricky.
    * At a certain point, there should be a timing cutoff to prevent a long string of unbroken speech for being too much for the model. At that point though, you likely didn't want to pass this to the model anyways.