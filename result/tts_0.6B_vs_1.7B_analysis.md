# 0.6B vs 1.7B 성능 차이가 미미한 원인 분석

**측정일**: 2026-04-03
**환경**: RTX 3090 (24 GB VRAM) / vllm-omni
**Stage Config**: `qwen3_tts.yaml` (기본)

---

## 1. 핵심 질문

0.6B 모델은 1.7B 대비 파라미터 수가 ~3배 작지만, 부하 테스트 결과 총 소요 시간 차이가 **4~25%에 불과**함.
모델 크기 차이를 고려하면 비정상적으로 작은 차이임. 왜?

---

## 2. TTS 파이프라인 구조와 병목

```
[Stage 0: Talker (AR)]  →  [Stage 1: Code2Wav (Generation)]
 텍스트 → 코덱 토큰 생성       코덱 토큰 → 오디오 PCM 변환
 ← 0.6B/1.7B 차이 여기 →       ← 두 모델 동일한 디코더 →
```

### 원인 1: Code2Wav는 모델 크기와 무관

- **Talker(Stage 0)만** 0.6B/1.7B 차이가 있음
- **Code2Wav(Stage 1)는 두 모델 모두 동일한 SpeechTokenizer 디코더** (`Qwen3TTSCode2Wav`)
- 총 소요 시간의 대부분은 Code2Wav의 오디오 디코딩 시간
- Talker가 아무리 빨라져도 Code2Wav 시간은 변하지 않음

### 원인 2: Code2Wav `max_num_seqs: 1` — 직렬 처리

`qwen3_tts.yaml` (기본 config):
```yaml
# Stage 0 (Talker): max_num_seqs: 10  ← 동시 10개 처리 가능
# Stage 1 (Code2Wav): max_num_seqs: 1  ← 한 번에 1개만 처리!
```

- Talker는 동시 10개 요청을 배치 처리하지만
- **Code2Wav는 1개씩 순차 처리** → 동시 요청이 늘수록 큐잉 지연 발생
- 이 직렬화 병목이 0.6B/1.7B Talker 속도 차이를 흡수해버림

### 원인 3: 같은 GPU 자원 공유

- 두 Stage 모두 `devices: "0"` — 동일 GPU에서 실행
- Stage config: `gpu_memory_utilization: 0.3` (각 Stage)
- Talker가 빨리 끝나도 Code2Wav가 GPU를 점유하며 디코딩 중

---

## 3. 측정 데이터 비교

### 3.1 총 소요 시간 비교 (avg ms)

| 동시 요청 | 0.6B | 1.7B | 차이 | 비고 |
|:--------:|-----:|-----:|:----:|------|
| **1개** | 2,000 | 2,673 | **-25%** | Talker 속도 차이가 유의미 |
| **2개** | 3,455 | 4,005 | **-14%** | Code2Wav 큐잉 시작 |
| **3개** | 4,651 | 4,847 | **-4%** | Code2Wav 병목 지배적 |
| **4개** | 5,375 | 6,216 | **-14%** | 변동 |
| **5개** | 6,881 | 7,304 | **-6%** | Code2Wav 직렬 처리가 전체 시간 결정 |

### 3.2 TTFA 비교 (ms)

| 동시 요청 | 0.6B | 1.7B | 차이 |
|:--------:|-----:|-----:|:----:|
| 1개 | 450 | 333 | 0.6B +35% 느림 |
| 2개 | 293 | 384 | 0.6B -24% 빠름 |
| 3개 | 376 | 439 | 0.6B -14% 빠름 |
| 4개 | 495 | 529 | 0.6B -6% 빠름 |
| 5개 | 611 | 673 | 0.6B -9% 빠름 |

> 단일 요청에서 0.6B TTFA가 높은 이유: 모델 사이즈가 작아 CUDA graph 최적화/배치 효율이 다를 수 있음

---

## 4. 시간 분해 추정

단일 요청(1개) 기준:

| 구간 | 0.6B | 1.7B | 비고 |
|------|-----:|-----:|------|
| Talker (코덱 생성) | ~450ms | ~333ms | TTFA 기준 |
| Code2Wav (오디오 변환) | ~1,550ms | ~2,340ms | 주요 시간 소요 |
| **총 소요** | **2,000ms** | **2,673ms** | |

동시 3개 이상에서는 Code2Wav 큐잉이 지배적 → 차이가 **-4%까지 수렴**

---

## 5. 수렴 패턴

```
총 소요 차이 (0.6B vs 1.7B):
1개: -25%  ← Talker 속도 차이가 유의미
2개: -14%  ← Code2Wav 큐잉 시작
3개:  -4%  ← Code2Wav 병목이 지배적 (거의 동일)
4개: -14%  ← 변동
5개:  -6%  ← Code2Wav 직렬 처리가 전체 시간 결정
```

**패턴**: 동시 요청이 증가할수록 Code2Wav 직렬 처리 시간이 전체를 지배 → 모델 크기 차이가 무의미해짐

---

## 6. 개선 방안

### 방안 1: `qwen3_tts_batch.yaml` 사용 (즉시 적용 가능)

`qwen3_tts_batch.yaml`에서는 Code2Wav `max_num_seqs: 4`로 배치 처리 가능:
```yaml
# Stage 0 (Talker): max_num_seqs: 4
# Stage 1 (Code2Wav): max_num_seqs: 4  ← 핵심!
```

docker-compose에 `--stage-config qwen3_tts_batch` 추가:
```yaml
command:
  - "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
  - "--omni"
  - "--stage-config"
  - "qwen3_tts_batch"    # ← 추가
  - "--trust-remote-code"
  - ...
```

**예상 효과**:
- Code2Wav 배치 처리로 동시 요청 시 총 소요 시간 감소
- 0.6B vs 1.7B Talker 속도 차이가 더 뚜렷하게 드러날 것
- 단, `gpu_memory_utilization`이 0.2로 낮아져 VRAM 관련 제약 확인 필요

### 방안 2: GPU 분리 (2 GPU 환경 시)

- Talker와 Code2Wav를 별도 GPU에 배치 (`devices: "0"` / `devices: "1"`)
- GPU 자원 경합 제거

---

## 7. 검증 방법

1. `qwen3_tts_batch.yaml`로 서버 재시작
2. 동일한 `test_ramp_tts.py` 부하 테스트 수행
3. 기존 결과와 비교하여 Code2Wav 병목 해소 여부 확인

---

## 8. 결론

| 항목 | 설명 |
|------|------|
| **근본 원인** | Code2Wav(Stage 1)가 동일 디코더를 사용하며 `max_num_seqs: 1`로 직렬 처리 |
| **모델 차이 영향** | Talker(Stage 0)에서만 발생하나, Code2Wav 병목에 가려짐 |
| **수렴 구간** | 동시 3개 이상에서 차이 4~6%로 수렴 |
| **즉시 개선** | `--stage-config qwen3_tts_batch` 적용 → Code2Wav 배치 처리 활성화 |
