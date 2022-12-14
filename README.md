# LiveSpeechToText
Live speech to text transcription.

# Method:

Super janky method.  
Start recording n-second samples.  
After every n seconds, save the audio as a .wav file.  
Send that audio to the model to translate to text.  
Delete the audio file.  
Repeat.  

# Notes:
When you run the first time you might get an error about "ffmpeg".  
This library is needed from by the transformers, but doesn't technically show up as a referenced library with the typical import.

```
conda install ffmpeg
```

You should see this as part of the virtual environment (SpeechToText).

# Problem Notes:
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
