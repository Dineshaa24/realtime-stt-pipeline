"""Audio helpers: PCM conversion, resampling and a telephone-channel simulation."""
from math import gcd

import numpy as np
from scipy.signal import resample_poly

TARGET_RATE = 16000  # what Whisper and the VAD expect


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def resample(samples: np.ndarray, src_rate: int, dst_rate: int = TARGET_RATE) -> np.ndarray:
    if src_rate == dst_rate:
        return samples.astype(np.float32)
    g = gcd(src_rate, dst_rate)
    return resample_poly(samples, dst_rate // g, src_rate // g).astype(np.float32)


def to_mono(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=1) if samples.ndim == 2 else samples


def simulate_telephony(samples: np.ndarray, rate: int = TARGET_RATE) -> np.ndarray:
    """Narrowband phone line: down to 8 kHz (drops everything above ~4 kHz), G.711 mu-law
    companding, then back up to 16 kHz the way a call-centre pipeline feeds Whisper."""
    narrow = resample(samples, rate, 8000)
    mu = 255.0
    encoded = np.sign(narrow) * np.log1p(mu * np.abs(narrow)) / np.log1p(mu)
    quantised = np.round((encoded + 1) * 127.5) / 127.5 - 1  # 8-bit codes
    decoded = np.sign(quantised) * (np.power(1 + mu, np.abs(quantised)) - 1) / mu
    return resample(decoded.astype(np.float32), 8000, rate)
