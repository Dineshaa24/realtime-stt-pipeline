"""Whisper transcription with faster-whisper (CTranslate2), int8 on CPU by default."""
import os
from typing import Protocol

import numpy as np


class Transcriber(Protocol):
    def transcribe(self, samples: np.ndarray) -> str: ...


class WhisperTranscriber:
    def __init__(self, model_size: str | None = None, device: str | None = None,
                 compute_type: str | None = None, beam_size: int | None = None):
        from faster_whisper import WhisperModel
        self.model_size = model_size or os.getenv("WHISPER_MODEL", "base.en")
        self.beam_size = beam_size or int(os.getenv("WHISPER_BEAM_SIZE", "1"))
        self.model = WhisperModel(self.model_size,
                                  device=device or os.getenv("WHISPER_DEVICE", "cpu"),
                                  compute_type=compute_type or os.getenv("WHISPER_COMPUTE", "int8"))

    def transcribe(self, samples: np.ndarray) -> str:
        # The VAD has already cut the audio into utterances, so Whisper's own VAD stays off.
        segments, _ = self.model.transcribe(samples, beam_size=self.beam_size, language="en",
                                            vad_filter=False, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()
