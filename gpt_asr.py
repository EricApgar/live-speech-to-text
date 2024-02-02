import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import sys


# Load the pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Function to transcribe speech
def recognize_speech(recording):
    # Convert the NumPy array of the recording to the right format
    input_values = processor(recording, return_tensors="pt", padding="longest").input_values

    # Perform inference with the model
    with torch.no_grad():
        logits = model(input_values).logits

    # Extract the predicted tokens
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the predicted tokens to text
    transcription = processor.decode(predicted_ids[0])

    return transcription

# Callback function for the audio stream
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)

    # Transcribe the speech
    transcription = recognize_speech(np.squeeze(indata))

    # Print the transcription
    print(transcription)

    # Stop if "stop" is detected
    if "stop" in transcription.lower():
        raise sd.CallbackStop

# Open the audio stream
with sd.InputStream(callback=callback):
    print("Speak now...")
    # Keep the stream open
    sd.sleep(1000000)
