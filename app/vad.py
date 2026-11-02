"""Streaming voice activity detection that cuts a live audio stream into utterances.

Audio arrives in arbitrary-sized chunks. It is re-framed into 30 ms frames, each frame is
classified as speech or not, and a small state machine decides where an utterance starts
and ends. A short pre-roll is kept so the first syllable is not clipped, and very long
utterances are force-cut so latency stays bounded.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from app.audio import TARGET_RATE, float_to_pcm16

FRAME_MS = 30
FRAME_SAMPLES = TARGET_RATE * FRAME_MS // 1000


@dataclass
class Segment:
    start: float  # seconds from the start of the stream
    end: float
    samples: np.ndarray


def webrtc_vad(aggressiveness: int = 2) -> Callable[[np.ndarray], bool]:
    import webrtcvad
    vad = webrtcvad.Vad(aggressiveness)
    return lambda frame: vad.is_speech(float_to_pcm16(frame), TARGET_RATE)


class Segmenter:
    def __init__(self, is_speech: Callable[[np.ndarray], bool], start_ms: int = 150,
                 end_silence_ms: int = 600, preroll_ms: int = 300, max_segment_s: float = 15.0):
        self.is_speech = is_speech
        self.start_frames = start_ms // FRAME_MS
        self.end_frames = end_silence_ms // FRAME_MS
        self.max_frames = int(max_segment_s * 1000 // FRAME_MS)
        self._preroll: deque[np.ndarray] = deque(maxlen=preroll_ms // FRAME_MS)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._frame_index = 0
        self._active: list[np.ndarray] | None = None
        self._start_frame = 0
        self._voiced_run = 0
        self._silent_run = 0

    def feed(self, samples: np.ndarray) -> Iterator[Segment]:
        """Add audio (16 kHz float32) and yield any utterances that just finished."""
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32)])
        while len(self._buffer) >= FRAME_SAMPLES:
            frame, self._buffer = self._buffer[:FRAME_SAMPLES], self._buffer[FRAME_SAMPLES:]
            segment = self._step(frame, self.is_speech(frame))
            self._frame_index += 1
            if segment is not None:
                yield segment

    def flush(self) -> Segment | None:
        """End of stream: return whatever utterance is still open."""
        return self._close(self._frame_index) if self._active else None

    def _step(self, frame: np.ndarray, speech: bool) -> Segment | None:
        if self._active is None:
            self._preroll.append(frame)
            self._voiced_run = self._voiced_run + 1 if speech else 0
            if self._voiced_run >= self.start_frames:
                self._active = list(self._preroll)
                self._start_frame = self._frame_index + 1 - len(self._preroll)
                self._preroll.clear()
                self._silent_run = 0
            return None

        self._active.append(frame)
        self._silent_run = 0 if speech else self._silent_run + 1
        if self._silent_run >= self.end_frames:
            # Drop most of the trailing silence but keep a little so words are not cut.
            keep = len(self._active) - self._silent_run + 3
            self._active = self._active[:keep]
            return self._close(self._start_frame + keep)
        if len(self._active) >= self.max_frames:
            return self._close(self._frame_index + 1)
        return None

    def _close(self, end_frame: int) -> Segment:
        samples = np.concatenate(self._active)
        segment = Segment(self._start_frame * FRAME_MS / 1000, end_frame * FRAME_MS / 1000, samples)
        self._active = None
        self._voiced_run = 0
        self._silent_run = 0
        return segment
