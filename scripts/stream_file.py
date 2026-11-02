"""Stream an audio file to the WebSocket endpoint at real-time speed and print transcripts as they arrive.

Usage: python -m scripts.stream_file call.wav [--url ws://localhost:8000/ws/stream] [--rate 8000]
"""
import argparse
import asyncio
import json
import time

import soundfile as sf
import websockets

from app.audio import float_to_pcm16, resample, to_mono


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--url", default="ws://localhost:8000/ws/stream")
    ap.add_argument("--rate", type=int, default=16000, help="sample rate to send (8000 = phone audio)")
    ap.add_argument("--chunk-ms", type=int, default=100)
    args = ap.parse_args()

    samples, rate = sf.read(args.path, dtype="float32")
    pcm = float_to_pcm16(resample(to_mono(samples), rate, args.rate))
    step = args.rate * args.chunk_ms // 1000 * 2

    async with websockets.connect(f"{args.url}?sample_rate={args.rate}") as ws:
        async def send():
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step])
                await asyncio.sleep(args.chunk_ms / 1000)  # pace like a live call
            await ws.send("end")

        sender = asyncio.create_task(send())
        start = time.perf_counter()
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "done":
                break
            print(f"[{time.perf_counter() - start:6.2f}s] {msg['start']:6.2f}-{msg['end']:6.2f}  {msg['text']}")
        await sender


if __name__ == "__main__":
    asyncio.run(main())
