"""Word error rate and speed on LibriSpeech clips, for clean 16 kHz audio and a simulated phone line.

Uses the 73-utterance LibriSpeech validation sample published as
hf-internal-testing/librispeech_asr_dummy. Two pipelines are compared:
  - whole clip straight into Whisper
  - streaming: the clip is fed through the VAD segmenter in 100 ms chunks, as the WebSocket does

Usage: python -m scripts.evaluate [--model base.en]
"""
import argparse
import io
import json
import re
import time
import urllib.request
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf

from app.audio import TARGET_RATE, resample, simulate_telephony, to_mono
from app.transcriber import WhisperTranscriber
from app.vad import Segmenter, webrtc_vad

DATA_URL = ("https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/"
            "resolve/main/clean/validation-00000-of-00001.parquet")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()


def load(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    clips = []
    for r in rows:
        samples, rate = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        clips.append((resample(to_mono(samples), rate), r["text"]))
    return clips


def streaming_transcribe(model, audio):
    seg = Segmenter(webrtc_vad(2))
    chunk = TARGET_RATE // 10
    parts = []
    for i in range(0, len(audio), chunk):
        parts += [model.transcribe(s.samples) for s in seg.feed(audio[i:i + chunk])]
    if (last := seg.flush()) is not None:
        parts.append(model.transcribe(last.samples))
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/librispeech_dummy.parquet")
    ap.add_argument("--model", default="base.en")
    args = ap.parse_args()

    clips = load(Path(args.data))
    audio_seconds = sum(len(a) for a, _ in clips) / TARGET_RATE
    model = WhisperTranscriber(model_size=args.model)
    model.transcribe(clips[0][0])  # warm-up

    refs = [normalise(t) for _, t in clips]
    results = {}
    for name, channel, streaming in [("clean_16k", None, False), ("telephone_8k", simulate_telephony, False),
                                     ("clean_16k_streaming", None, True),
                                     ("telephone_8k_streaming", simulate_telephony, True)]:
        hyps, start = [], time.perf_counter()
        for audio, _ in clips:
            audio = channel(audio) if channel else audio
            hyps.append(streaming_transcribe(model, audio) if streaming else model.transcribe(audio))
        took = time.perf_counter() - start
        results[name] = {"wer": round(jiwer.wer(refs, [normalise(h) for h in hyps]), 4),
                         "real_time_factor": round(took / audio_seconds, 3)}
        print(name, results[name])

    out = {"model": args.model, "compute": "int8 CPU", "clips": len(clips),
           "audio_minutes": round(audio_seconds / 60, 2), "results": results}
    Path("results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
