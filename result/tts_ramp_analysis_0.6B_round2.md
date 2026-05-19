# TTS 단계적 부하 테스트 분석 — 0.6B Round 2 (Ramp 1→2→3→4→5→6→7)

**측정일**: 2026-04-08
**서버**: RTX 3090 (24 GB VRAM) / vllm-omni Docker
**모델**: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` · Voice: `femail_achernar`
**테스트**: `tests/test_ramp_tts.py` — 단계 완료 후 3초 sleep, 다음 단계 시작

---

## 1. TTS 요청 결과

### 1.1 개별 요청 상세

| 단계 | # | TTFA (ms) | 총 소요 (ms) | 오디오 길이 | 파일 |
|:---:|:--:|----------:|------------:|----------:|------|
| **1개** | 0 | 450 | 2,000 | 4.4s | ramp_n1_0.wav |
| **2개** | 0 | 271 | 3,059 | 5.8s | ramp_n2_0.wav |
|  | 1 | 315 | 3,851 | 9.0s | ramp_n2_1.wav |
| **3개** | 0 | 428 | 3,932 | 6.0s | ramp_n3_0.wav |
|  | 1 | 326 | 5,175 | 9.8s | ramp_n3_1.wav |
|  | 2 | 374 | 4,846 | 8.6s | ramp_n3_2.wav |
| **4개** | 0 | 397 | 3,827 | 5.0s | ramp_n4_0.wav |
|  | 1 | 447 | 5,979 | 9.4s | ramp_n4_1.wav |
|  | 2 | 480 | 5,505 | 8.3s | ramp_n4_2.wav |
|  | 3 | 658 | 6,191 | 10.2s | ramp_n4_3.wav |
| **5개** | 0 | 508 | 5,225 | 5.7s | ramp_n5_0.wav |
|  | 1 | 451 | 7,257 | 9.5s | ramp_n5_1.wav |
|  | 2 | 552 | 6,998 | 8.7s | ramp_n5_2.wav |
|  | 3 | 784 | 7,553 | 10.3s | ramp_n5_3.wav |
|  | 4 | 760 | 7,372 | 9.4s | ramp_n5_4.wav |
| **6개** | 0~5 | ERR | ~145,445 | 실패 | peer closed connection without sending |
| **7개** | 0~6 | ERR | ~65 | 실패 | [Errno 61] Connection refused |

### 1.2 단계별 평균 (1개 기준 증가율)

| 동시 요청 | avg TTFA (ms) | TTFA 증감 | avg 총 소요 (ms) | 총 소요 증가 | 성공률 | 벽시계 (ms) |
|:--------:|-------------:|:--------:|---------------:|:---------:|:----:|----------:|
| **1개** | 450 | — | 2,000 | — | 1/1 | 2,003 |
| **2개** | 293 | **-35%** | 3,455 | +73% | 2/2 | 3,855 |
| **3개** | 376 | -16% | 4,651 | +133% | 3/3 | 5,178 |
| **4개** | 495 | +10% | 5,375 | +169% | 4/4 | 6,194 |
| **5개** | 611 | +36% | 6,881 | +244% | 5/5 | 7,558 |
| **6개** | 실패 | — | 실패 | — | 0/6 | 145,449 |
| **7개** | 실패 | — | 실패 | — | 0/7 | 73 |

---

## 2. 6개 동시 요청 서버 Crash 분석

### 2.1 결정적 증거: 서버 로그

```
(Worker pid=150) INFO 04-08 02:04:36 [multiproc_executor.py:759] Parent process exited, terminating worker queues
(Worker pid=150) INFO 04-08 02:04:36 [multiproc_executor.py:854] WorkerProc shutting down.
```

**Worker pid=150 (Talker, Stage-0)**이 부모 프로세스(Engine Core) 사망을 감지하고 종료.
6개 요청이 등록된 지 **~1초 만에** crash 발생.

### 2.2 프로세스 사망 순서

```
APIServer (pid=1)
  └── Engine Core (parent)          ← ★ 1. 먼저 사망 (segfault/OOM 추정)
        ├── Worker pid=150 (Talker)  ← 2. 부모 사망 감지 → graceful shutdown
        └── Worker pid=340 (Code2Wav)← 3. 살아있으나 입력 없음 → hang
```

1. **Engine Core** (Talker의 부모 프로세스)가 6개 요청 스케줄링 중 crash
2. **Worker pid=150** (Talker)이 `Parent process exited` 감지 → 자발적 종료
3. **Worker pid=340** (Code2Wav)은 살아있으나, Talker로부터 코덱 토큰을 받지 못해 **hang**
4. 클라이언트 6개 연결이 ~145초간 대기 후 타임아웃 → `peer closed connection`
5. 7개 단계는 서버가 이미 down → `[Errno 61] Connection refused`

### 2.3 GPU 로그 — VRAM 해제 증거

| 시간 (KST) | 이벤트 | VRAM | GPU util |
|:---:|------|-----:|--------:|
| 11:04:35 | 6개 요청 등록 | **16,048 MiB** | 0% |
| 11:04:36 | Worker pid=150 종료 시작 | 16,048 MiB | 0% |
| 11:04:37 | 마지막 GPU 활동 (잔여 처리) | 16,048 MiB | **62%** |
| **11:04:38.5** | **Talker VRAM 해제** | **→ 7,993 MiB** | 0% |
| 11:04:39+ | 완전 idle (Code2Wav만 잔존) | 7,993 MiB | 0% |
| 11:05:00+ | 전력 idle (23W) | 7,993 MiB | 0% |

**VRAM 16,048 → 7,993 MiB = -8,055 MiB 감소**

```
VRAM 구성 (crash 전):
  Talker  (pid=150): ~8,055 MiB  ← crash 시 해제됨
  Code2Wav (pid=340): ~7,993 MiB  ← 유지 (살아있으나 hang)
  합계:               ~16,048 MiB
```

### 2.4 Crash 원인 추정

서버 로그에 Python traceback이 없음 → Engine Core가 **segfault 또는 OOM으로 즉사** (graceful error 아님)

| 후보 | 설명 |
|------|------|
| **누적 상태 오버플로우** | 1→2→3→4→5개 단계에서 총 15개 요청 처리 후 내부 상태(SharedMemoryConnector, 스케줄러 큐) 미정리 상태에서 6개 추가 |
| **KV cache 부족** | docker-compose `--max-model-len 1024` (stage config 기본값 4096 대비 1/4) — 6개 배치 시 KV cache 부족 가능 |
| **Code2Wav malformed 경고 누적** | 매 단계 완료 시 `input_ids length 1 not divisible by num_quantizers 16` 경고가 N회 발생 — 내부 상태 오염 가능 |

---

## 3. GPU 부하 분석 (정상 단계 1~5개)

### 3.1 단계별 GPU 활성 구간

| 단계 | GPU 활성 구간 (KST) | 지속 | max util | max power | VRAM |
|:---:|:------------|:---:|--------:|----------:|-----:|
| 1개 | 11:03:57 ~ 11:03:58 | **~2s** | 91% | 265W | 16,044 MiB |
| 2개 | 11:04:02 ~ 11:04:05 | **~4s** | 98% | 313W | 16,046 MiB |
| 3개 | 11:04:09 ~ 11:04:13 | **~5s** | 99% | 319W | 16,046 MiB |
| 4개 | 11:04:17 ~ 11:04:22 | **~6s** | 100% | 319W | 16,048 MiB |
| 5개 | 11:04:26 ~ 11:04:32 | **~7s** | 100% | 320W | 16,048 MiB |
| 6개 | 11:04:37 (1s만) | **~1s** | 62% | 208W | 16,048→**7,993** |

### 3.2 GPU Utilization 패턴

각 단계는 **고 util → 저 util** 2단계 구조:
- **Talker 구동 구간**: 86~100% (코덱 토큰 생성, 고부하)
- **Code2Wav 구동 구간**: 33~75% (오디오 디코딩, 중부하)

```
1개  ███████████████████████████████████████████ 91%→58%           ~2s
2개  ██████████████████████████████████████████████████ 94%→98%→48% ~4s
3개  ████████████████████████████████████████████████████████ 91%→99%→42%→70% ~5s
4개  ██████████████████████████████████████████████████████████████ 88%→100%→33%→75% ~6s
5개  ████████████████████████████████████████████████████████████████████ 86%→100%→34%→38% ~7s
```

### 3.3 전력 소비 추이 (0.5s 평균, W)

```
1개: 191 → 265 → 231 → 212 → 171                               (단일 burst)
2개: 143 → 268 → 313 → 274 → 239 → 233 → 230 → 208           (2s peak + 2s tail)
3개: 138 → 231 → 318 → 319 → 299 → 267 → 230 → 223 → 233 → 224  (3s peak + 2s tail)
4개: 128 → 221 → 310 → 319 → 319 → 284 → 258 → 235 → 233 → 225 → 232 → 234  (3s peak + 3s tail)
5개: 133 → 246 → 311 → 320 → 319 → 319 → 269 → 252 → 239 → 214 → 244 → 241 → 227 → 239  (4s peak + 3s tail)
```

- **320W 피크**: 3개 이상에서 TDP 근처 도달
- **중반대 하락**: Code2Wav 디코딩 구간에서 220~260W로 하강

---

## 4. Code2Wav malformed 경고 분석

매 단계 완료 시 반복되는 WARNING:

```
(Worker pid=340) WARNING: Code2Wav input_ids length 1 not divisible by num_quantizers 16;
                          skipping malformed request.
```

| 단계 | 경고 횟수 | 발생 시점 |
|:---:|:-------:|----------|
| 1개 | 1회 | 02:03:58 |
| 2개 | 2회 | 02:04:04~05 |
| 3개 | 3회 | 02:04:11~13 |
| 4개 | 4회 | 02:04:19~22 |
| 5개 | 5회 | 02:04:30~32 |

- 각 요청의 **end-of-stream 시그널**(길이 1)이 Code2Wav에서 malformed로 처리되는 패턴
- 오디오 생성 자체는 성공하지만, 요청 수에 비례하여 경고 누적
- 5개 단계까지 총 15회 경고 → 내부 상태 오염이 6개 단계 crash의 간접 원인 가능성

---

## 5. Real-Time Factor (RTF) 분석

> RTF = 오디오 길이 / 총 소요 시간. 1.0 미만이면 실시간 처리 불가.

| 단계 | 최장 오디오 | 해당 요청 총 소요 | RTF | 여유 |
|:---:|----------:|:--------------:|:---:|:---:|
| 1개 | 4.4s | 2,000ms | **2.20x** | 충분 |
| 2개 | 9.0s | 3,851ms | **2.34x** | 충분 |
| 3개 | 9.8s | 5,175ms | **1.89x** | 양호 |
| 4개 | 10.2s | 6,191ms | **1.65x** | 주의 |
| 5개 | 10.3s | 7,553ms | **1.36x** | 경계 |

---

## 6. 1.7B 모델 비교

### 6.1 TTFA 비교 (ms)

| 동시 요청 | 0.6B | 1.7B | 차이 |
|:--------:|-----:|-----:|:----:|
| 1개 | 450 | 333 | 0.6B +35% 느림 |
| 2개 | 293 | 384 | 0.6B -24% 빠름 |
| 3개 | 376 | 439 | 0.6B -14% 빠름 |
| 4개 | 495 | 529 | 0.6B -6% 빠름 |
| 5개 | 611 | 673 | 0.6B -9% 빠름 |

### 6.2 총 소요 비교 (avg ms)

| 동시 요청 | 0.6B | 1.7B | 차이 |
|:--------:|-----:|-----:|:----:|
| 1개 | 2,000 | 2,673 | 0.6B **-25%** 빠름 |
| 2개 | 3,455 | 4,005 | 0.6B **-14%** 빠름 |
| 3개 | 4,651 | 4,847 | 0.6B **-4%** 빠름 |
| 4개 | 5,375 | 6,216 | 0.6B **-14%** 빠름 |
| 5개 | 6,881 | 7,304 | 0.6B **-6%** 빠름 |

### 6.3 성능 차이가 작은 근본 원인

```
TTS 파이프라인:
  [Stage 0: Talker (AR)]  →  [Stage 1: Code2Wav (Generation)]
   텍스트 → 코덱 토큰 생성       코덱 토큰 → 오디오 PCM 변환
   ← 0.6B/1.7B 차이 여기 →       ← 두 모델 동일한 디코더 →
```

| 원인 | 상세 |
|------|------|
| **Code2Wav 동일** | 0.6B/1.7B 모두 같은 SpeechTokenizer 디코더 사용. 총 소요 시간의 대부분을 차지 |
| **Code2Wav max_num_seqs=1** | 기본 config에서 Code2Wav는 한 번에 1개만 처리 → 직렬 병목 |
| **동시 요청 증가 시 수렴** | 1개 -25% → 3개 -4% → Code2Wav 직렬 처리가 전체 시간을 지배하면서 모델 크기 차이 흡수 |

---

## 7. 종합 결론

### 7.1 운영 권장 기준 (RTX 3090 단일 GPU, 0.6B 모델)

| 구분 | 동시 요청 수 | 근거 |
|------|:-----------:|------|
| 최적 (안정+효율) | **2~3개** | TTFA 최저(-35%~-16%), RTF 1.89x 이상 |
| 허용 최대 | **4~5개** | 성공률 100%, RTF 1.36x 이상 |
| 금지 | **6개 이상** | Engine Core crash 확인 (Worker pid=150 즉사) |

### 7.2 Crash 방지를 위한 개선 방안

| 방안 | 설명 |
|------|------|
| `--max-model-len` 확대 | 현재 1024 → 4096으로 변경하여 KV cache 여유 확보 |
| `qwen3_tts_batch.yaml` 사용 | Code2Wav `max_num_seqs: 4`로 배치 처리 활성화 |
| 서버 재시작 간격 설정 | 누적 malformed 경고에 의한 상태 오염 방지 |
| `--max-num-seqs` 제한 | Talker 동시 처리 수를 5개로 제한하여 6개 이상 방지 |

---

## 8. 측정 환경

```
GPU:         NVIDIA GeForce RTX 3090 (24,576 MiB VRAM, TDP 350W)
Docker:      docker/docker-compose.yml
모델:        Qwen/Qwen3-TTS-12Hz-0.6B-Base
설정:        --max-model-len 1024, --gpu-memory-utilization 0.9, --dtype bfloat16
Stage config: qwen3_tts.yaml (기본값, Code2Wav max_num_seqs=1)
GPU 모니터링: nvidia-smi --query-gpu=... -l 0.1 → result/gpu_log_0.6B_round2.csv
테스트 파일:  tests/test_ramp_tts.py
단계:         [1, 2, 3, 4, 5, 6, 7]
단계 간 sleep: 3초
서버 로그:    result/server_log_0408.txt
결과 파일:    result/multi_tts_rusult_0.6B_round2.txt
```

### 8.1 Docker 설정

```yaml
services:
  vllm-qwen3-tts:
    image: vllm-omni:custom
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - VLLM_USE_V1=1
      - VLLM_ATTENTION_BACKEND=FLASHINFER
      - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    command:
      - "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
      - "--omni"
      - "--trust-remote-code"
      - "--max-model-len"
      - "1024"
      - "--dtype"
      - "bfloat16"
      - "--gpu-memory-utilization"
      - "0.9"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "30000"
```
