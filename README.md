# realtime-stt-pipeline

Streaming speech-to-text with Whisper. Audio is sent over a WebSocket in small chunks, a voice activity detector cuts it into utterances as they finish, and each utterance comes back as text a moment after the speaker stops. There is also a plain file-upload endpoint.

It handles 8 kHz phone audio as well as 16 kHz, because most of the speech I work with comes from contact-centre calls, not studio microphones.

## How it works

```
client ──PCM chunks──▶ WebSocket ──▶ resample to 16 kHz ──▶ VAD segmenter ──▶ Whisper ──▶ {"type": "final", "text": ...}
```

- **VAD segmenter** (`app/vad.py`): re-frames the stream into 30 ms frames and runs WebRTC VAD on each. An utterance starts after 150 ms of speech and ends after 600 ms of silence. A 300 ms pre-roll stops the first syllable from being clipped, and anything longer than 15 s is force-cut so latency stays bounded. The result is the same whatever chunk size the client sends, which is covered by tests.
- **Whisper** (`app/transcriber.py`): [faster-whisper](https://github.com/SYSTRAN/faster-whisper), `base.en`, int8 on CPU by default. Model, device and precision are set through environment variables, so the same code runs `large-v3` on a GPU.
- **Concurrency**: decodes run in a thread pool behind a semaphore (`MAX_CONCURRENT_TRANSCRIPTIONS`), so one busy stream can't starve the others.
- **Phone-line simulation** (`app/audio.py`): down to 8 kHz, G.711 mu-law companding, and back up to 16 kHz. This is what the evaluation uses to measure the cost of telephone audio.

## Results

73 LibriSpeech validation utterances (8 minutes of read English), Whisper `base.en`, int8, greedy decoding, 8-core laptop CPU:

| Pipeline | WER | Real-time factor |
|---|---|---|
| Clean 16 kHz, whole clip | 8.9% | 0.39 |
| Phone line 8 kHz, whole clip | 11.5% | 0.51 |
| Clean 16 kHz, streaming through the VAD | 9.0% | 0.43 |
| Phone line 8 kHz, streaming through the VAD | 11.4% | 0.52 |

A real-time factor of 0.39 means one second of audio takes 0.39 seconds to transcribe, so a single CPU worker keeps up with a live stream with room to spare.

What I took from this:

- Cutting the stream with the VAD costs almost nothing in accuracy (9.0% vs 8.9%), so streaming is not a trade-off here.
- Narrowband phone audio adds about 2.5 points of WER, because Whisper was trained mostly on wideband audio. That gap is why fine-tuning on real 8 kHz call audio (or using a telephony-trained model) matters in production.
- `base.en` is a small model chosen to run on CPU. Bigger models cut WER noticeably but need a GPU to stay real-time.

Reproduce with `python -m scripts.evaluate --model base.en`.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app
```

Stream a file the way a live call would arrive (paced at real time, here as 8 kHz phone audio):

```bash
python -m scripts.stream_file call.wav --rate 8000
```

```
[  9.32s]   0.39-  5.97  Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
[ 17.41s]   6.81- 11.46  Nor is Mr. Quilter's manner less interesting than his matter.
```

Or with Docker (the image downloads the model during the build):

```bash
docker build -t realtime-stt-pipeline .
docker run -p 8000:8000 realtime-stt-pipeline
```

## API

**WebSocket** `ws://host:8000/ws/stream?sample_rate=8000`

Send raw 16-bit mono PCM as binary messages, then the text message `end`. For each utterance you receive:

```json
{"type": "final", "start": 6.81, "end": 11.46, "text": "Nor is Mr. Quilter's manner less interesting than his matter.", "latency_ms": 1840}
```

and `{"type": "done"}` at the end. `latency_ms` is the decode time once the utterance has ended. The end-of-speech wait (600 ms of silence) comes on top of it.

**POST** `/transcribe`: upload any file `libsndfile` can read (wav, flac, ogg). Returns the full text, timed segments, duration and real-time factor.

**GET** `/health`

## Tests

```bash
pytest
```

17 tests covering the VAD state machine (boundaries, chunk-size independence, force-cuts, ignoring clicks), audio conversion and the phone-line filter, and the HTTP and WebSocket endpoints. They use a fake VAD and a fake transcriber, so they run in seconds without downloading a model.

## Configuration

| Variable | Default | |
|---|---|---|
| `WHISPER_MODEL` | `base.en` | any faster-whisper model, e.g. `small.en`, `large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cuda` for GPU |
| `WHISPER_COMPUTE` | `int8` | `float16` on GPU |
| `WHISPER_BEAM_SIZE` | `1` | greedy by default for latency |
| `VAD_AGGRESSIVENESS` | `2` | 0-3, higher filters more noise |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | `2` | parallel decodes |

## What I'd do next

- Partial (interim) results while someone is still speaking, by decoding the open segment every second or so.
- Speaker diarization for two-party calls, so agent and customer are labelled separately.
- Swap WebRTC VAD for Silero VAD, which handles background noise in call centres better.
- Benchmark NVIDIA Parakeet against Whisper on the same 8 kHz set.

## License

MIT
