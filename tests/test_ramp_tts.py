"""
TTS 단계적 부하 테스트 — 1 → 2 → 3 → 4 → 5개 순차 실행

동작 순서:
  1단계(1개) 완료 → 3초 sleep → 2단계(2개) 완료 → 3초 sleep → ... → 5단계(5개) 완료
  각 단계 완료 시 결과 테이블 출력

Usage:
    python tests/test_ramp_tts.py
    python tests/test_ramp_tts.py --steps 1 3 5        # 특정 단계만
    python tests/test_ramp_tts.py --sleep 5            # 단계 간 sleep 5초
    python tests/test_ramp_tts.py --same-text          # 모든 요청에 동일 텍스트

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

DEFAULT_STEPS = [1, 2, 3, 4, 5]
DEFAULT_SLEEP_S = 3

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


@dataclass
class StepSummary:
    n: int
    wall_ms: float
    results: list[TTSResult]

    @property
    def ok(self) -> list[TTSResult]:
        return [r for r in self.results if r.success]

    @property
    def avg_ttfa(self) -> float:
        return sum(r.ttfa_ms for r in self.ok) / len(self.ok) if self.ok else 0.0

    @property
    def avg_total(self) -> float:
        return sum(r.total_ms for r in self.ok) / len(self.ok) if self.ok else 0.0


# ---------------------------------------------------------------------------
# 단일 요청 (스레드 내 실행)
# ---------------------------------------------------------------------------

def fetch_one(idx: int, text: str, step_n: int, barrier: threading.Barrier) -> TTSResult:
    result = TTSResult(idx=idx, text_preview=text[:20])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = OUT_DIR / f"ramp_n{step_n}_{idx}.wav"
    result.wav_path = wav_path

    payload = {
        "model": MODEL,
        "input": text,
        "voice": VOICE_NAME,
        "response_format": "pcm",
        "stream": True,
    }

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
# 단계 실행
# ---------------------------------------------------------------------------

def run_step(n: int, same_text: bool) -> StepSummary:
    texts = [TEXTS[0] if same_text else TEXTS[i % len(TEXTS)] for i in range(n)]
    barrier = threading.Barrier(n)
    results: list[TTSResult | None] = [None] * n

    def worker(idx: int) -> None:
        results[idx] = fetch_one(idx, texts[idx], n, barrier)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]

    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_ms = (time.perf_counter() - wall_start) * 1000

    final = [r for r in results if r is not None]
    return StepSummary(n=n, wall_ms=wall_ms, results=final)


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------

SEP = "─" * 70

def print_step(summary: StepSummary) -> None:
    n = summary.n
    print(f"\n{'━' * 70}")
    print(f"  단계: 동시 {n}개 요청  |  전체 소요: {summary.wall_ms:,.0f} ms")
    print(f"{'━' * 70}")
    print(f" {'#':>2} | {'TTFA (ms)':>9} | {'총 소요 (ms)':>12} | {'오디오 길이':>10} | 파일")
    print(SEP)

    for r in summary.results:
        if r.success:
            print(
                f" {r.idx:>2} | {r.ttfa_ms:>9,.0f} | {r.total_ms:>12,.0f} |"
                f" {r.duration_s:>8.1f}s  | {r.wav_path.name}"
            )
        else:
            print(f" {r.idx:>2} | {'ERR':>9} | {r.total_ms:>12,.0f} | {'실패':>10} | {r.error[:38]}")

    print(SEP)
    ok = summary.ok
    if ok:
        print(f"평균 TTFA: {summary.avg_ttfa:,.0f} ms  |  평균 총 소요: {summary.avg_total:,.0f} ms  |  성공: {len(ok)}/{n}")
    else:
        print("모든 요청 실패")


def print_final_summary(summaries: list[StepSummary]) -> None:
    print(f"\n\n{'=' * 70}")
    print("  전체 단계 요약")
    print(f"{'=' * 70}")
    print(f" {'동시 요청':>6} | {'avg TTFA (ms)':>13} | {'avg 총 소요 (ms)':>16} | {'성공률':>6} | 벽시계 (ms)")
    print("─" * 70)

    base_ttfa = summaries[0].avg_ttfa if summaries else 1
    base_total = summaries[0].avg_total if summaries else 1

    for s in summaries:
        if s.ok:
            ttfa_diff = f"+{(s.avg_ttfa / base_ttfa - 1) * 100:.0f}%" if s.n > summaries[0].n else "  기준"
            total_diff = f"+{(s.avg_total / base_total - 1) * 100:.0f}%" if s.n > summaries[0].n else "  기준"
            print(
                f" {s.n:>6}개  | {s.avg_ttfa:>9,.0f}  ({ttfa_diff:>6}) | "
                f"{s.avg_total:>12,.0f}  ({total_diff:>6}) | "
                f"{len(s.ok)}/{s.n:>2}   | {s.wall_ms:,.0f}"
            )
        else:
            print(f" {s.n:>6}개  | {'실패':>13} | {'실패':>16} | {0}/{s.n:>2}   | {s.wall_ms:,.0f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS 단계적 부하 테스트 (1→2→3→4→5개)")
    parser.add_argument(
        "--steps", type=int, nargs="+", default=DEFAULT_STEPS,
        metavar="N", help=f"실행할 동시 요청 수 목록 (기본값: {DEFAULT_STEPS})"
    )
    parser.add_argument(
        "--sleep", type=float, default=DEFAULT_SLEEP_S,
        help=f"단계 간 대기 시간(초) (기본값: {DEFAULT_SLEEP_S}s)"
    )
    parser.add_argument(
        "--same-text", action="store_true", help="모든 요청에 동일 텍스트 사용"
    )
    args = parser.parse_args()

    steps: list[int] = args.steps
    sleep_s: float = args.sleep
    same_text: bool = args.same_text

    print(f"단계: {steps}  |  단계 간 sleep: {sleep_s}s")

    summaries: list[StepSummary] = []

    for i, n in enumerate(steps):
        summary = run_step(n, same_text)
        print_step(summary)
        summaries.append(summary)

        if i < len(steps) - 1:
            print(f"\n  ▷ {sleep_s:.0f}초 대기 후 다음 단계({steps[i+1]}개) 시작...")
            time.sleep(sleep_s)

    print_final_summary(summaries)
