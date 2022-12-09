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