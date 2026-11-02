import numpy as np
import pytest

from app.vad import Segmenter
from tests.conftest import energy_vad, silence, tone


def run(audio, chunk, **kw):
    seg = Segmenter(energy_vad, **kw)
    out = []
    for i in range(0, len(audio), chunk):
        out += list(seg.feed(audio[i:i + chunk]))
    if (last := seg.flush()) is not None:
        out.append(last)
    return out


def test_finds_both_utterances_with_sensible_boundaries(two_utterances):
    segs = run(two_utterances, chunk=1600)
    assert len(segs) == 2
    # pre-roll starts a little before the speech, end trims most of the trailing silence
    assert segs[0].start == pytest.approx(0.85, abs=0.2)
    assert segs[0].end == pytest.approx(2.1, abs=0.15)
    assert segs[1].start == pytest.approx(2.85, abs=0.2)
    assert segs[1].end == pytest.approx(3.6, abs=0.15)
    assert all(len(s.samples) > 0 for s in segs)


@pytest.mark.parametrize("chunk", [160, 777, 4096, 48000])
def test_chunk_size_does_not_change_the_result(two_utterances, chunk):
    reference = [(s.start, s.end) for s in run(two_utterances, chunk=1600)]
    assert [(s.start, s.end) for s in run(two_utterances, chunk=chunk)] == reference


def test_long_speech_is_force_cut():
    segs = run(tone(7.0), chunk=1600, max_segment_s=3.0)
    assert len(segs) == 3
    assert all(s.end - s.start <= 3.0 + 1e-6 for s in segs)


def test_short_click_is_ignored():
    audio = np.concatenate([silence(0.5), tone(0.06), silence(1.0)])  # 60 ms < 150 ms start threshold
    assert run(audio, chunk=1600) == []


def test_open_utterance_is_returned_on_flush():
    segs = run(np.concatenate([silence(0.5), tone(1.0)]), chunk=1600)
    assert len(segs) == 1 and segs[0].end == pytest.approx(1.5, abs=0.05)
