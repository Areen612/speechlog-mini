from faster_whisper import WhisperModel


class ASRService:
    def __init__(self, model_size: str = "base"):
        # device="cpu" is simplest and stable for interview demo
        self.model = WhisperModel(model_size, device="cpu")

    def transcribe_segment(self, audio_segment):
        # faster-whisper accepts numpy float32 arrays
        segments, info = self.model.transcribe(audio_segment, language="ar")
        text = " ".join([s.text.strip() for s in segments])
        return text