from faster_whisper import WhisperModel

model = WhisperModel("tiny.en")

segments, info = model.transcribe("audio.flac")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
