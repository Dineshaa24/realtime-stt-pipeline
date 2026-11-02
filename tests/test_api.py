import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.audio import float_to_pcm16, resample
from tests.conftest import RATE, energy_vad


class FakeTranscriber:
    def transcribe(self, samples):
        return f"utterance of {len(samples) / RATE:.1f} seconds"


@pytest.fixture
def client(monkeypatch):
    from app import main
    monkeypatch.setattr(main, "build_transcriber", FakeTranscriber)
    monkeypatch.setattr(main, "build_vad", lambda: energy_vad)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_transcribe_file(client, two_utterances):
    buf = io.BytesIO()
    sf.write(buf, two_utterances, RATE, format="WAV")
    body = client.post("/transcribe", files={"file": ("a.wav", buf.getvalue(), "audio/wav")}).json()
    assert len(body["segments"]) == 2
    assert body["duration_s"] == pytest.approx(4.5)
    assert body["text"].startswith("utterance of")


def test_transcribe_rejects_garbage(client):
    r = client.post("/transcribe", files={"file": ("a.wav", b"not audio", "audio/wav")})
    assert r.status_code == 415


@pytest.mark.parametrize("rate", [16000, 8000])
def test_websocket_stream(client, two_utterances, rate):
    pcm = float_to_pcm16(resample(two_utterances, RATE, rate))
    chunk = rate // 10 * 2  # 100 ms of 16-bit audio
    with client.websocket_connect(f"/ws/stream?sample_rate={rate}") as ws:
        for i in range(0, len(pcm), chunk):
            ws.send_bytes(pcm[i:i + chunk])
        ws.send_text("end")
        messages = []
        while (m := ws.receive_json())["type"] != "done":
            messages.append(m)
    assert [m["type"] for m in messages] == ["final", "final"]
    assert messages[0]["start"] < messages[0]["end"] < messages[1]["start"]


def test_websocket_rejects_bad_rate(client):
    with client.websocket_connect("/ws/stream?sample_rate=12345") as ws:
        with pytest.raises(Exception):
            ws.receive_json()
