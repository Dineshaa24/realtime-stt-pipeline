import numpy as np
import pytest

RATE = 16000


def tone(seconds: float, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def energy_vad(frame: np.ndarray) -> bool:
    """Deterministic stand-in for WebRTC VAD: loud frame = speech."""
    return float(np.abs(frame).max()) > 0.1


@pytest.fixture
def two_utterances() -> np.ndarray:
    # speech at 1.0-2.0 s and 3.0-3.5 s
    return np.concatenate([silence(1.0), tone(1.0), silence(1.0), tone(0.5), silence(1.0)])
