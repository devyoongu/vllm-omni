"""
Smoke test for a running Qwen3-TTS-12Hz-1.7B-Base server.

Targets the already-running Docker service; does NOT spin up its own server.

Usage:
    python tests/test_qwen3_tts_live.py                        # default: localhost:30000
    python tests/test_qwen3_tts_live.py --host 192.168.0.10 --port 30000
    pytest tests/test_qwen3_tts_live.py -v

Output audio files are saved to /tmp/tts_test_*.wav.
"""

import argparse
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_HOST = "http://172.31.79.202"
DEFAULT_PORT = 30000
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Reference audio for voice-clone test (publicly accessible)
REF_AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)

OUTPUT_DIR = Path("/tmp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def base_url(host: str, port: int) -> str:
    # Strip scheme if the user included it in host (e.g. "http://172.31.79.202")
    for prefix in ("https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix):]
    host = host.rstrip("/")
    # If host already contains a port (e.g. "172.31.79.202:30000"), use it as-is
    if ":" in host:
        return f"http://{host}"
    return f"http://{host}:{port}"


def ok(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  [PASS] {label}{suffix}")


def fail(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  [FAIL] {label}{suffix}")
    sys.exit(1)


def assert_wav_bytes(data: bytes, label: str) -> None:
    """Minimal WAV header check: first 4 bytes must be 'RIFF'."""
    if len(data) < 44 or data[:4] != b"RIFF":
        fail(label, f"response is not a valid WAV (got {len(data)} bytes, header={data[:8]!r})")


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------


def test_health(host: str, port: int) -> None:
    url = f"{base_url(host, port)}/health"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        ok("GET /health", f"status={r.status_code}")
    else:
        fail("GET /health", f"status={r.status_code}")


def test_models(host: str, port: int) -> None:
    url = f"{base_url(host, port)}/v1/models"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        fail("GET /v1/models", f"status={r.status_code}")
    data = r.json()
    model_ids = [m["id"] for m in data.get("data", [])]
    if not model_ids:
        fail("GET /v1/models", "empty model list")
    ok("GET /v1/models", f"models={model_ids}")


def test_tts_basic(host: str, port: int) -> None:
    """Single non-streaming TTS request. Base model requires task_type + ref_audio/ref_text."""
    url = f"{base_url(host, port)}/v1/audio/speech"
    payload = {
        "model": MODEL,
        "input": "Hello, this is a test of the text to speech system.",
        "response_format": "wav",
        "voice": "clone",
        "task_type": "Base",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=120)
    elapsed = time.perf_counter() - t0

    if r.status_code != 200:
        fail("POST /v1/audio/speech (basic)", f"status={r.status_code}, body={r.text[:300]}")

    audio = r.content
    assert_wav_bytes(audio, "POST /v1/audio/speech (basic)")

    out = OUTPUT_DIR / "tts_test_basic.wav"
    out.write_bytes(audio)
    ok("POST /v1/audio/speech (basic)", f"{len(audio):,} bytes in {elapsed:.1f}s → {out}")


def test_tts_voice_clone(host: str, port: int) -> None:
    """TTS with voice cloning using a reference audio URL."""
    url = f"{base_url(host, port)}/v1/audio/speech"
    payload = {
        "model": MODEL,
        "input": "The weather is nice today, perfect for a walk in the park.",
        "response_format": "wav",
        "voice": "clone",
        "extra_body": {
            "task_type": "Base",
            "ref_audio": REF_AUDIO_URL,
            "ref_text": REF_TEXT,
        },
    }
    # extra_body must be a top-level key for vllm-omni, not nested
    flat_payload = {
        "model": MODEL,
        "input": "The weather is nice today, perfect for a walk in the park.",
        "response_format": "wav",
        "voice": "clone",
        "task_type": "Base",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=flat_payload, timeout=120)
    elapsed = time.perf_counter() - t0

    if r.status_code != 200:
        fail("POST /v1/audio/speech (clone)", f"status={r.status_code}, body={r.text[:300]}")

    audio = r.content
    assert_wav_bytes(audio, "POST /v1/audio/speech (clone)")

    out = OUTPUT_DIR / "tts_test_clone.wav"
    out.write_bytes(audio)
    ok("POST /v1/audio/speech (clone)", f"{len(audio):,} bytes in {elapsed:.1f}s → {out}")


def test_tts_streaming(host: str, port: int) -> None:
    """Streaming TTS request: verifies chunks arrive and concatenated audio is valid WAV."""
    url = f"{base_url(host, port)}/v1/audio/speech"
    payload = {
        "model": MODEL,
        "input": "Streaming audio generation test.",
        "response_format": "wav",
        "stream": True,
        "voice": "clone",
        "task_type": "Base",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    t0 = time.perf_counter()
    chunks = []
    with requests.post(url, json=payload, stream=True, timeout=120) as r:
        if r.status_code != 200:
            fail("POST /v1/audio/speech (stream)", f"status={r.status_code}, body={r.text[:300]}")
        for chunk in r.iter_content(chunk_size=4096):
            if chunk:
                chunks.append(chunk)
    elapsed = time.perf_counter() - t0

    if not chunks:
        fail("POST /v1/audio/speech (stream)", "received 0 chunks")

    audio = b"".join(chunks)
    assert_wav_bytes(audio, "POST /v1/audio/speech (stream)")

    out = OUTPUT_DIR / "tts_test_stream.wav"
    out.write_bytes(audio)
    ok(
        "POST /v1/audio/speech (stream)",
        f"{len(chunks)} chunks, {len(audio):,} bytes in {elapsed:.1f}s → {out}",
    )


# ---------------------------------------------------------------------------
# pytest entry points (optional)
# ---------------------------------------------------------------------------

import os
_HOST = os.environ.get("TTS_HOST", DEFAULT_HOST)
_PORT = int(os.environ.get("TTS_PORT", DEFAULT_PORT))


def test_health_pytest():
    test_health(_HOST, _PORT)


def test_models_pytest():
    test_models(_HOST, _PORT)


def test_tts_basic_pytest():
    test_tts_basic(_HOST, _PORT)


def test_tts_voice_clone_pytest():
    test_tts_voice_clone(_HOST, _PORT)


def test_tts_streaming_pytest():
    test_tts_streaming(_HOST, _PORT)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for running Qwen3-TTS server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    print(f"\nTarget: {base_url(args.host, args.port)}")
    print(f"Model:  {MODEL}\n")

    test_health(args.host, args.port)
    test_models(args.host, args.port)
    test_tts_basic(args.host, args.port)
    test_tts_voice_clone(args.host, args.port)
    test_tts_streaming(args.host, args.port)

    print("\nAll tests passed.")
