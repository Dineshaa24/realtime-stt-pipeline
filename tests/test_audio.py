import numpy as np

from app.audio import float_to_pcm16, pcm16_to_float, resample, simulate_telephony
from tests.conftest import RATE, tone


def test_pcm_round_trip():
    x = tone(0.1)
    assert np.allclose(pcm16_to_float(float_to_pcm16(x)), x, atol=1e-4)


def test_resample_lengths():
    x = tone(1.0)
    assert len(resample(x, RATE, 8000)) == 8000
    assert len(resample(resample(x, RATE, 8000), 8000, RATE)) == RATE
    assert len(resample(np.zeros(44100, np.float32), 44100)) == RATE


def _energy_above(x, hz):
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / RATE)
    return spectrum[freqs > hz].sum() / spectrum.sum()


def test_telephony_removes_high_frequencies():
    x = tone(1.0, freq=6000) + tone(1.0, freq=500)
    assert _energy_above(x, 4000) > 0.4
    assert _energy_above(simulate_telephony(x), 4000) < 0.01
    assert len(simulate_telephony(x)) == len(x)
