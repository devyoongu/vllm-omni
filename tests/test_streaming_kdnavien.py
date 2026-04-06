"""
실시간 스트리밍 TTS 재생 테스트 — vllm-omni Qwen3-TTS-12Hz-1.7B-Base

동작 순서:
  1. wav/femail_achernar.wav 를 /v1/audio/voices API로 서버에 등록
  2. 등록한 voice 이름으로 /v1/audio/speech 스트리밍 요청
  3. PCM 청크 수신 즉시 스피커 재생 + TTFA 측정
  4. 전체 오디오를 WAV 파일로 저장

Requirements:
    pip install httpx sounddevice
"""

import queue
import threading
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

SERVER = "http://172.31.79.202:30000"
MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

VOICE_NAME = "femail_achernar"
REF_AUDIO_PATH = Path(__file__).parent.parent / "wav" / "femail_achernar.wav"

INPUT_TEXT = (
    "이 에러는 stream_player_put_frame()에 전달된 pending_data의 크기가 PJSIP가 허용하는 최대 프레임 크기를 초과했다는 의미입니다."
)

SAMPLE_RATE = 24_000  # Qwen3-TTS Code2Wav 고정 출력
CHANNELS = 1
OUTPUT_PATH = Path("/tmp/kdnavien_tts_output.wav")



# ---------------------------------------------------------------------------
# Step 2: 스트리밍 TTS 재생
# ---------------------------------------------------------------------------

def stream_and_play() -> None:
    payload = {
        "model": MODEL,
        "input": INPUT_TEXT,
        "voice": VOICE_NAME,
        "response_format": "pcm",
        "stream": True,
    }

    print(f"\n서버: {SERVER}")
    print(f"Voice: {VOICE_NAME}")
    print(f"입력: {INPUT_TEXT[:50]}...")
    print("-" * 55)

    # --- 네트워크 수신 스레드 → 큐 → 오디오 재생 스레드 ---
    audio_q: queue.Queue[bytes | None] = queue.Queue(maxsize=64)
    pcm_frames = bytearray()
    ttfa_ms_ref: list[float] = []
    start_time = time.perf_counter()

    def fetch() -> None:
        """서버에서 PCM 청크를 받아 큐에 넣는다."""
        leftover = bytearray()
        try:
            with httpx.stream(
                "POST",
                f"{SERVER}/v1/audio/speech",
                json=payload,
                timeout=httpx.Timeout(None),
            ) as response:
                if response.status_code != 200:
                    body = response.read().decode()
                    print(f"[ERROR] status={response.status_code}\n{body}")
                    audio_q.put(None)
                    return

                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    if not ttfa_ms_ref:
                        ttfa_ms_ref.append((time.perf_counter() - start_time) * 1000)

                    leftover.extend(chunk)
                    aligned_len = (len(leftover) // 2) * 2
                    if aligned_len:
                        audio_q.put(bytes(leftover[:aligned_len]))
                        del leftover[:aligned_len]

                if leftover:
                    audio_q.put(bytes(leftover))
        except Exception as e:
            print(f"[ERROR fetch] {e}")
        finally:
            audio_q.put(None)  # 종료 신호

    fetch_thread = threading.Thread(target=fetch, daemon=True)
    fetch_thread.start()

    # 오디오 디바이스: 큐에서 꺼내 일정한 속도로 재생
    with sd.RawOutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=2400,          # 100ms 단위 블록 — 언더런 방지
    ) as audio_stream:
        first_chunk = True
        while True:
            chunk = audio_q.get()
            if chunk is None:
                break

            if first_chunk:
                print(f"첫 청크 수신 — TTFA: {ttfa_ms_ref[0]:.0f} ms" if ttfa_ms_ref else "첫 청크 수신")
                print("-" * 55)
                first_chunk = False

            audio_stream.write(chunk)
            pcm_frames.extend(chunk)

    fetch_thread.join()

    total_ms = (time.perf_counter() - start_time) * 1000
    duration_s = len(pcm_frames) / (SAMPLE_RATE * 2 * CHANNELS)
    print(f"\n재생 완료 — 총 소요: {total_ms:.0f} ms | 오디오 길이: {duration_s:.1f}s")

    if pcm_frames:
        with wave.open(str(OUTPUT_PATH), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(pcm_frames))
        print(f"WAV 저장: {OUTPUT_PATH}  ({len(pcm_frames):,} bytes)")
        print(f"로컬 재생: open {OUTPUT_PATH}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # voice는 미리 등록되어 있어야 합니다.
    # 등록 방법:
    #   curl -X POST http://172.31.79.202:30000/v1/audio/voices \
    #     -F "name=femail_achernar" \
    #     -F "consent=agreed" \
    #     -F "ref_text=안녕하세요 메타엠 시연용 콜봇입니다 무엇을 도와드릴까요" \
    #     -F "audio_sample=@/Users/yglee/python/vllm-omni/wav/femail_achernar.wav"
    stream_and_play()
