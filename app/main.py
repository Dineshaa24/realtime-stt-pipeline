"""FastAPI service: live transcription over WebSocket and one-shot file transcription."""
import asyncio
import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.audio import TARGET_RATE, pcm16_to_float, resample, to_mono
from app.vad import Segmenter, webrtc_vad

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "2"))
state: dict = {}


def build_transcriber():
    """Load the Whisper model. Tests replace this with a fake."""
    from app.transcriber import WhisperTranscriber
    return WhisperTranscriber()


def build_vad():
    return webrtc_vad(int(os.getenv("VAD_AGGRESSIVENESS", "2")))


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["transcriber"] = build_transcriber()
    # Whisper is CPU/GPU bound; cap parallel decodes so one busy stream can't starve the rest.
    state["slots"] = asyncio.Semaphore(MAX_CONCURRENT)
    yield
    state.clear()


app = FastAPI(title="Real-time speech-to-text", version="1.0.0", lifespan=lifespan)


async def transcribe(samples: np.ndarray) -> tuple[str, float]:
    async with state["slots"]:
        start = time.perf_counter()
        text = await run_in_threadpool(state["transcriber"].transcribe, samples)
        return text, time.perf_counter() - start


@app.get("/health")
def health():
    return {"status": "ok", "max_concurrent": MAX_CONCURRENT}


@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    try:
        samples, rate = sf.read(io.BytesIO(await file.read()), dtype="float32")
    except Exception:
        raise HTTPException(status_code=415, detail="unsupported or corrupt audio file")
    audio = resample(to_mono(samples), rate)
    duration = len(audio) / TARGET_RATE

    segmenter = Segmenter(build_vad())
    pieces = list(segmenter.feed(audio))
    if (last := segmenter.flush()) is not None:
        pieces.append(last)

    results, compute = [], 0.0
    for seg in pieces:
        text, took = await transcribe(seg.samples)
        compute += took
        if text:
            results.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
    return {
        "text": " ".join(r["text"] for r in results),
        "segments": results,
        "duration_s": round(duration, 2),
        "real_time_factor": round(compute / duration, 3) if duration else None,
    }


@app.websocket("/ws/stream")
async def stream(ws: WebSocket, sample_rate: int = TARGET_RATE):
    """Client sends raw 16-bit mono PCM as binary messages and the text "end" when done.
    Server replies with {"type": "final", "start", "end", "text", "latency_ms"} per utterance."""
    await ws.accept()
    if sample_rate not in (8000, 16000, 44100, 48000):
        await ws.close(code=1003, reason="unsupported sample_rate")
        return
    segmenter = Segmenter(build_vad())

    async def emit(seg):
        text, took = await transcribe(seg.samples)
        if text:
            await ws.send_json({"type": "final", "start": round(seg.start, 2), "end": round(seg.end, 2),
                                "text": text, "latency_ms": round(took * 1000)})

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes"):
                audio = resample(pcm16_to_float(message["bytes"]), sample_rate)
                for seg in segmenter.feed(audio):
                    await emit(seg)
            elif message.get("text") == "end":
                if (last := segmenter.flush()) is not None:
                    await emit(last)
                await ws.send_json({"type": "done"})
                await ws.close()
                return
    except WebSocketDisconnect:
        return
