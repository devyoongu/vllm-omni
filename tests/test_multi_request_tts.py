"""
다중 TTS 요청 동시 실행 테스트 — vllm-omni Qwen3-TTS-12Hz-1.7B-Base

동작 순서:
  1. threading.Barrier 로 N개 스레드를 동시에 출발
  2. 각 스레드가 /v1/audio/speech 스트리밍 요청 → PCM 수집
  3. TTFA·총 소요·오디오 길이 집계 후 요약 테이블 출력
  4. WAV 파일 /tmp/tts_{idx}.wav 저장 (재생 없음)

Usage:
    python tests/test_multi_request_tts.py              # 기본 3개 동시 요청
    python tests/test_multi_request_tts.py --n 5        # 5개 동시 요청
    python tests/test_multi_request_tts.py --n 10 --same-text  # 동일 텍스트 10개

Requirements:
    pip install httpx
"""

import argparse
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

SERVER = "http://172.31.79.202:30000"
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICE_NAME = "femail_achernar"

SAMPLE_RATE = 24_000
CHANNELS = 1

OUT_DIR = Path(__file__).parent.parent / "out_wav"

# 10개 입력 텍스트 (N > 10 시 순환)
TEXTS = [
    "안녕하세요, 경동나비엔 고객센터입니다. 무엇을 도와드릴까요?",
    "이 에러는 stream_player_put_frame()에 전달된 pending_data의 크기가 PJSIP가 허용하는 최대 프레임 크기를 초과했다는 의미입니다.",
    "현재 서울 지역의 날씨는 맑음이며, 기온은 섭씨 22도입니다. 외출 시 가벼운 겉옷을 준비하세요.",
    "고객님의 예약이 확인되었습니다. 예약 번호는 A-1-2-3-4-5입니다. 방문 전 확인 문자를 드리겠습니다.",
    "안내 말씀드립니다. 5월 1일 근로자의 날은 고객센터 운영이 중단됩니다. 긴급 문의는 앱을 이용해 주세요.",
    "보일러 온도를 올리고 싶으신가요? 리모컨의 온도 버튼을 길게 누르시면 설정 화면으로 이동합니다.",
    "고객님, 현재 접수하신 AS 신청은 내일 오전 10시에 기사님이 방문 예정입니다. 일정 변경이 필요하시면 말씀해 주세요.",
    "제품 보증 기간은 구매일로부터 2년이며, 소모품은 보증 대상에서 제외됩니다. 자세한 내용은 제품 설명서를 참고해 주세요.",
    "난방 효율을 높이려면 외출 모드보다 온도를 낮게 유지하는 것이 더 효과적입니다. 설정 온도를 18도로 낮춰보시겠어요?",
    "경동나비엔 멤버십에 가입하시면 정기 점검 서비스와 부품 할인 혜택을 받으실 수 있습니다. 지금 바로 앱에서 가입해 보세요.",
]


# ---------------------------------------------------------------------------
# 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class TTSResult:
    idx: int
    text_preview: str
    ttfa_ms: float = 0.0
    total_ms: float = 0.0
    duration_s: float = 0.0
    wav_path: Path = field(default_factory=Path)
    success: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# 단일 요청 (스레드 내 실행)
# ---------------------------------------------------------------------------

def fetch_one(idx: int, text: str, barrier: threading.Barrier) -> TTSResult:
    result = TTSResult(idx=idx, text_preview=text[:20])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = OUT_DIR / f"tts_{idx}.wav"
    result.wav_path = wav_path

    payload = {
        "model": MODEL,
        "input": text,
        "voice": VOICE_NAME,
        "response_format": "pcm",
        "stream": True,
    }

    # 모든 스레드가 준비되면 동시에 출발
    barrier.wait()
    start = time.perf_counter()

    ttfa_recorded = False
    pcm_frames = bytearray()
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
                result.error = f"status={response.status_code}: {body[:200]}"
                result.total_ms = (time.perf_counter() - start) * 1000
                return result

            for chunk in response.iter_bytes():
                if not chunk:
                    continue

                if not ttfa_recorded:
                    result.ttfa_ms = (time.perf_counter() - start) * 1000
                    ttfa_recorded = True

                leftover.extend(chunk)
                aligned_len = (len(leftover) // 2) * 2
                if aligned_len:
                    pcm_frames.extend(leftover[:aligned_len])
                    del leftover[:aligned_len]

            if leftover:
                pcm_frames.extend(leftover)

    except Exception as exc:
        result.error = str(exc)
        result.total_ms = (time.perf_counter() - start) * 1000
        return result

    result.total_ms = (time.perf_counter() - start) * 1000
    result.duration_s = len(pcm_frames) / (SAMPLE_RATE * 2 * CHANNELS)

    if pcm_frames:
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(pcm_frames))
        result.success = True

    return result


# ---------------------------------------------------------------------------
# 동시 실행
# ---------------------------------------------------------------------------

def run_concurrent(n: int, same_text: bool = False) -> list[TTSResult]:
    texts = [TEXTS[0] if same_text else TEXTS[i % len(TEXTS)] for i in range(n)]
    barrier = threading.Barrier(n)
    results: list[TTSResult | None] = [None] * n

    def worker(idx: int) -> None:
        results[idx] = fetch_one(idx, texts[idx], barrier)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]

    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_ms = (time.perf_counter() - wall_start) * 1000

    final: list[TTSResult] = [r for r in results if r is not None]
    print_summary(final, wall_ms)
    return final


# ---------------------------------------------------------------------------
# 요약 출력
# ---------------------------------------------------------------------------

def print_summary(results: list[TTSResult], wall_ms: float) -> None:
    n = len(results)
    sep = "─" * 65

    print(f"\n동시 요청 수: {n}  |  전체 소요: {wall_ms:,.0f} ms")
    print(sep)
    print(f" {'#':>2} | {'TTFA (ms)':>9} | {'총 소요 (ms)':>12} | {'오디오 길이':>10} | 파일")
    print("─" * 4 + "|" + "─" * 11 + "|" + "─" * 14 + "|" + "─" * 12 + "|" + "─" * 22)

    ok_results = [r for r in results if r.success]

    for r in results:
        if r.success:
            print(
                f" {r.idx:>2} | {r.ttfa_ms:>9,.0f} | {r.total_ms:>12,.0f} |"
                f" {r.duration_s:>8.1f}s  | {r.wav_path}"
            )
        else:
            print(f" {r.idx:>2} | {'ERR':>9} | {r.total_ms:>12,.0f} | {'실패':>10} | {r.error[:40]}")

    print(sep)

    if ok_results:
        avg_ttfa = sum(r.ttfa_ms for r in ok_results) / len(ok_results)
        avg_total = sum(r.total_ms for r in ok_results) / len(ok_results)
        print(f"평균 TTFA: {avg_ttfa:,.0f} ms  |  평균 총 소요: {avg_total:,.0f} ms")
        print(f"성공: {len(ok_results)}/{n}")
    else:
        print("모든 요청 실패")

    print()
    if ok_results:
        paths = " ".join(str(r.wav_path) for r in ok_results)
        print(f"로컬 재생: open {paths}")


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다중 TTS 동시 요청 테스트")
    parser.add_argument("--n", type=int, default=1, help="동시 요청 수 (기본값: 3)")
    parser.add_argument(
        "--same-text", action="store_true", help="모든 요청에 동일 텍스트 사용"
    )
    args = parser.parse_args()

    run_concurrent(n=args.n, same_text=args.same_text)
