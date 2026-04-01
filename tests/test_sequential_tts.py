"""
TTS 순차 처리 테스트 — 100개 문장을 하나씩 순서대로 처리

동작 순서:
  1. 100개 문장을 순차적으로 /v1/audio/speech 요청
  2. 각 요청 완료 후 out_wav/sequential/{번호:03d}_{미리보기}.wav 저장
  3. 전체 완료 후 요약 출력

Usage:
    python tests/test_sequential_tts.py
    python tests/test_sequential_tts.py --out-dir /tmp/tts_out
    python tests/test_sequential_tts.py --count 20        # 앞 20개만

Requirements:
    pip install httpx
"""

import argparse
import time
import wave
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

DEFAULT_OUT_DIR = Path(__file__).parent.parent / "out_wav" / "sequential"

# ---------------------------------------------------------------------------
# 100개 문장
# ---------------------------------------------------------------------------

SENTENCES = [
    # 인사 / 콜봇 오프닝
    "안녕하세요, 경동나비엔 고객센터입니다. 무엇을 도와드릴까요?",
    "감사합니다. 경동나비엔을 이용해 주셔서 감사합니다.",
    "잠시만 기다려 주세요. 상담원을 연결해 드리겠습니다.",
    "고객님, 통화 가능하신가요? 확인 후 다시 안내해 드리겠습니다.",
    "네, 말씀하세요. 고객님의 문의 사항을 도와드리겠습니다.",
    # 제품 안내
    "경동나비엔 콘덴싱 보일러는 열효율 98% 이상으로 에너지를 절감합니다.",
    "나비엔 온수매트는 전자파를 차단한 안전한 제품입니다.",
    "이 제품은 스마트폰 앱으로 원격 제어가 가능합니다.",
    "보일러 필터는 6개월마다 교체하시는 것을 권장합니다.",
    "온수기 설정 온도는 최대 섭씨 60도까지 가능합니다.",
    # 설치 안내
    "설치 기사님은 예약 후 평균 2영업일 이내에 방문합니다.",
    "설치 시 가스 공급이 정상인지 미리 확인해 주세요.",
    "배관 연결부에 누수가 없는지 설치 후 꼭 점검해 주세요.",
    "전기 콘센트는 반드시 접지 단자가 있는 것을 사용하세요.",
    "설치 완료 후 제품 등록을 하시면 보증 혜택을 받으실 수 있습니다.",
    # AS / 오류 대응
    "에러 코드 E1은 점화 실패를 의미합니다. 가스 밸브를 확인해 주세요.",
    "에러 코드 E3은 과열 방지 장치가 동작했다는 의미입니다.",
    "에러 코드 E7은 온도 센서 이상입니다. 서비스 센터에 문의하세요.",
    "보일러가 자동으로 꺼진다면 배기 막힘이 원인일 수 있습니다.",
    "물이 나오지 않는다면 단수 여부 또는 필터 막힘을 확인해 주세요.",
    # 오류 기술 설명
    "이 에러는 stream_player_put_frame에 전달된 데이터 크기가 초과됐다는 의미입니다.",
    "PJSIP 최대 프레임 크기를 초과하면 오디오 전송이 끊길 수 있습니다.",
    "pending_data 버퍼가 가득 찼을 때 이 오류가 발생합니다.",
    "해결 방법은 청크 크기를 줄이거나 전송 속도를 조절하는 것입니다.",
    "로그를 확인해 어느 시점에 버퍼 오버플로우가 발생하는지 파악하세요.",
    # 날씨 / 계절 안내
    "현재 서울 기온은 섭씨 22도이며 맑은 날씨가 예상됩니다.",
    "오늘 오후에는 소나기가 내릴 수 있으니 우산을 준비하세요.",
    "기온이 영하로 내려가면 보일러 동파 방지 기능을 켜두세요.",
    "한파 주의보 발령 시 외출 모드보다 실내 온도를 최소 10도로 설정하세요.",
    "여름철 보일러는 온수 전용 모드로 운영하는 것이 효율적입니다.",
    # 예약 / 일정
    "고객님의 AS 예약이 완료되었습니다. 내일 오전 10시에 기사님이 방문합니다.",
    "예약 번호는 A-1-2-3-4-5입니다. 꼭 메모해 두세요.",
    "방문 30분 전 기사님께서 먼저 연락드릴 예정입니다.",
    "일정 변경은 방문 하루 전까지 고객센터로 연락 주세요.",
    "오늘 예약 가능한 시간대는 오후 2시와 오후 4시입니다.",
    # 보증 / 멤버십
    "제품 보증 기간은 구매일로부터 2년입니다.",
    "소모품인 필터와 패킹은 보증 대상에서 제외됩니다.",
    "나비엔 멤버십에 가입하시면 정기 점검 서비스가 무료로 제공됩니다.",
    "멤버십 회원은 부품 교체 시 10% 할인 혜택을 받으십니다.",
    "앱에서 보증 등록을 하시면 만료일을 쉽게 확인하실 수 있습니다.",
    # 사용 팁
    "보일러 온도를 1도 낮추면 연료비를 약 3% 절약할 수 있습니다.",
    "외출 시에는 온도를 완전히 끄기보다 15도로 유지하는 것이 효율적입니다.",
    "온수 절약을 위해 샤워 시간을 1분 줄이면 연간 수만 원을 아낄 수 있습니다.",
    "리모컨의 예약 기능을 활용하면 취침 전 자동으로 온도를 낮출 수 있습니다.",
    "보일러 주변에 가연성 물질을 두지 마시고 환기를 유지해 주세요.",
    # 공지사항
    "5월 1일 근로자의 날은 고객센터 운영이 중단됩니다.",
    "긴급 수리는 앱 또는 홈페이지 야간 접수를 이용해 주세요.",
    "추석 연휴 기간 중 AS 접수는 제한적으로 운영됩니다.",
    "신제품 출시 기념으로 이달 말까지 설치비 할인 행사를 진행합니다.",
    "서비스 품질 향상을 위해 통화가 녹음될 수 있습니다.",
    # 가스 안전
    "가스 냄새가 나면 즉시 창문을 열고 가스 밸브를 잠그세요.",
    "전기 스위치나 콘센트를 절대 만지지 마세요.",
    "한국가스안전공사 긴급 신고 번호는 1544-4500입니다.",
    "가스 누출 시 화기를 절대 사용하지 마세요.",
    "정기적으로 가스 배관 점검을 받으시면 사고를 예방할 수 있습니다.",
    # 난방 관련
    "바닥 난방이 따뜻하지 않다면 공기 빼기 작업이 필요할 수 있습니다.",
    "난방수 압력 게이지가 1 이하이면 보충수를 추가하세요.",
    "배관 안에 공기가 차면 온수 순환이 원활하지 않을 수 있습니다.",
    "각방 밸브가 잠겨 있지 않은지 먼저 확인해 주세요.",
    "실내 온도와 체감 온도 차이가 크다면 기밀 시공 상태를 점검하세요.",
    # 앱 안내
    "나비엔 스마트 앱을 설치하면 외출 중에도 온도를 조절할 수 있습니다.",
    "앱에서 에너지 사용량을 주간, 월간 단위로 확인할 수 있습니다.",
    "알림 설정을 켜두시면 필터 교체 시기를 자동으로 안내받으실 수 있습니다.",
    "앱 계정 하나로 최대 세 대의 제품을 동시에 관리할 수 있습니다.",
    "앱 업데이트 후 재시작하시면 새로운 기능을 사용하실 수 있습니다.",
    # 환경 / 절약
    "콘덴싱 보일러는 일반 보일러보다 이산화탄소 배출을 30% 줄입니다.",
    "에너지 효율 1등급 제품을 사용하면 연간 난방비를 크게 절감할 수 있습니다.",
    "태양광 패널과 보일러를 함께 사용하면 에너지 자립도를 높일 수 있습니다.",
    "보일러 노후화 시 신형 교체로 에너지 낭비를 줄이는 것을 권장합니다.",
    "절전 모드를 활성화하면 대기 전력 소비를 최소화할 수 있습니다.",
    # 고객 응대
    "고객님 말씀 잘 들었습니다. 빠르게 처리해 드리겠습니다.",
    "불편을 드려서 진심으로 사과드립니다.",
    "더 궁금하신 점이 있으시면 언제든지 연락해 주세요.",
    "저희 서비스를 이용해 주셔서 감사합니다.",
    "오늘 상담이 도움이 되셨으면 좋겠습니다.",
    # 콜봇 오류 / 재시도
    "죄송합니다, 말씀을 잘 알아듣지 못했습니다. 다시 한번 말씀해 주시겠어요?",
    "네, 고객님. 방금 말씀하신 내용을 확인하겠습니다.",
    "잠시 후 다시 시도해 주시면 더 빠르게 도움을 드릴 수 있습니다.",
    "현재 상담 대기 인원이 많아 연결이 다소 지연될 수 있습니다.",
    "홈페이지 자주 묻는 질문 코너에서도 답변을 확인하실 수 있습니다.",
    # 납부 / 결제
    "서비스 요금은 방문 완료 후 현장에서 결제하실 수 있습니다.",
    "카드와 현금 모두 결제 가능하며 현금 영수증도 발급됩니다.",
    "정기 점검 서비스는 연 단위로 사전 결제하시면 10% 할인됩니다.",
    "부품 비용은 공식 홈페이지 가격표를 참고하세요.",
    "영수증은 문자 또는 이메일로 받으실 수 있습니다.",
    # 지역 / 센터
    "가장 가까운 서비스 센터는 앱 또는 홈페이지에서 찾으실 수 있습니다.",
    "서울 강남 센터는 평일 오전 9시부터 오후 6시까지 운영합니다.",
    "제주도 지역은 출장 기사 배정 시 추가 비용이 발생할 수 있습니다.",
    "전국 300여 개의 서비스 센터가 고객님을 기다리고 있습니다.",
    "도서 산간 지역은 방문까지 3일 이상 소요될 수 있습니다.",
    # 마무리
    "통화 주셔서 감사합니다. 오늘 하루도 좋은 하루 되세요.",
    "경동나비엔은 항상 고객님 곁에 있겠습니다.",
    "문의사항은 언제든 전화 또는 앱으로 연락 주세요.",
    "상담이 종료되었습니다. 감사합니다.",
    "고객님의 소중한 의견은 서비스 개선에 활용됩니다. 감사합니다.",
    # 추가
    "보일러 전원이 켜지지 않는다면 차단기가 내려가 있는지 확인해 주세요.",
    "온수가 갑자기 차가워진다면 온수 설정 온도를 다시 확인해 주세요.",
    "고객님의 제품 모델명은 제품 전면 스티커에서 확인하실 수 있습니다.",
    "서비스 접수 후 진행 상황은 앱 푸시 알림으로 실시간 안내됩니다.",
    "오늘 상담 내용은 고객님의 계정에 자동 저장되어 다음 문의 시 활용됩니다.",
]

assert len(SENTENCES) == 100, f"문장 수 오류: {len(SENTENCES)}"


# ---------------------------------------------------------------------------
# 단일 요청
# ---------------------------------------------------------------------------

def tts_one(idx: int, text: str, out_dir: Path) -> dict:
    """TTS 요청 1건 수행. 결과 dict 반환."""
    payload = {
        "model": MODEL,
        "input": text,
        "voice": VOICE_NAME,
        "response_format": "pcm",
        "stream": True,
    }

    # 파일명: 001_안녕하세요.wav (미리보기 10자, 안전한 문자만)
    preview = text[:10].replace(" ", "_").replace(",", "").replace(".", "")
    wav_path = out_dir / f"{idx:03d}_{preview}.wav"

    ttfa_ms = 0.0
    ttfa_recorded = False
    pcm_frames = bytearray()
    leftover = bytearray()

    start = time.perf_counter()

    try:
        with httpx.stream(
            "POST",
            f"{SERVER}/v1/audio/speech",
            json=payload,
            timeout=httpx.Timeout(None),
        ) as response:
            if response.status_code != 200:
                body = response.read().decode()
                elapsed = (time.perf_counter() - start) * 1000
                return {"idx": idx, "success": False, "error": f"HTTP {response.status_code}: {body[:100]}",
                        "ttfa_ms": 0, "total_ms": elapsed, "duration_s": 0, "wav_path": wav_path}

            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                if not ttfa_recorded:
                    ttfa_ms = (time.perf_counter() - start) * 1000
                    ttfa_recorded = True
                leftover.extend(chunk)
                aligned = (len(leftover) // 2) * 2
                if aligned:
                    pcm_frames.extend(leftover[:aligned])
                    del leftover[:aligned]

            if leftover:
                pcm_frames.extend(leftover)

    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return {"idx": idx, "success": False, "error": str(exc),
                "ttfa_ms": 0, "total_ms": elapsed, "duration_s": 0, "wav_path": wav_path}

    total_ms = (time.perf_counter() - start) * 1000
    duration_s = len(pcm_frames) / (SAMPLE_RATE * 2 * CHANNELS)

    if pcm_frames:
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(pcm_frames))

    return {
        "idx": idx,
        "success": bool(pcm_frames),
        "error": "",
        "ttfa_ms": ttfa_ms,
        "total_ms": total_ms,
        "duration_s": duration_s,
        "wav_path": wav_path,
    }


# ---------------------------------------------------------------------------
# 순차 실행
# ---------------------------------------------------------------------------

def run_sequential(sentences: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(sentences)
    results = []
    wall_start = time.perf_counter()

    print(f"출력 폴더: {out_dir}")
    print(f"총 문장: {total}개  |  순차 처리 시작\n")
    print(f" {'#':>3} | {'TTFA':>7} | {'총 소요':>8} | {'길이':>6} | 파일명")
    print("─" * 75)

    for i, text in enumerate(sentences):
        r = tts_one(i + 1, text, out_dir)
        results.append(r)

        if r["success"]:
            print(
                f" {r['idx']:>3} | {r['ttfa_ms']:>6.0f}ms | {r['total_ms']:>7.0f}ms"
                f" | {r['duration_s']:>5.1f}s | {r['wav_path'].name}"
            )
        else:
            print(f" {r['idx']:>3} | {'ERR':>7} | {r['total_ms']:>7.0f}ms | {'실패':>6} | {r['error'][:40]}")

    wall_ms = (time.perf_counter() - wall_start) * 1000

    # 요약
    ok = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]

    print("─" * 75)
    print(f"\n완료: {len(ok)}/{total}  |  실패: {len(fail)}  |  전체 소요: {wall_ms / 1000:.1f}s")

    if ok:
        avg_ttfa = sum(r["ttfa_ms"] for r in ok) / len(ok)
        avg_total = sum(r["total_ms"] for r in ok) / len(ok)
        avg_dur = sum(r["duration_s"] for r in ok) / len(ok)
        print(f"평균 TTFA: {avg_ttfa:.0f}ms  |  평균 총 소요: {avg_total:.0f}ms  |  평균 오디오 길이: {avg_dur:.1f}s")

    if fail:
        print(f"\n실패 목록:")
        for r in fail:
            print(f"  #{r['idx']:>3}: {r['error'][:60]}")

    print(f"\nWAV 파일 위치: {out_dir}")


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS 100개 문장 순차 처리")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"WAV 저장 폴더 (기본값: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--count", type=int, default=len(SENTENCES),
        help=f"처리할 문장 수 (기본값: {len(SENTENCES)})"
    )
    args = parser.parse_args()

    sentences = SENTENCES[: args.count]
    run_sequential(sentences, args.out_dir)
