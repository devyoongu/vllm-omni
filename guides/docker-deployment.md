# vllm-omni Docker 배포 가이드

Qwen3-TTS-12Hz-1.7B-Base 모델을 원격 GPU 서버에 Docker로 배포하는 절차입니다.

## 서버 환경

| 항목 | 값 |
|------|-----|
| OS | Ubuntu 18.04.5 LTS |
| GPU | NVIDIA RTX 3090 (24GB), GPU 0 |
| CUDA | 12.9 / Driver 575.57.08 |
| 기본 Python | 2.7 (시스템) → Docker로 우회 (아래 참고) |

### Python 환경 문제와 Docker 선택 이유

Ubuntu 18.04의 시스템 기본 Python은 **2.7.17**이며, `apt`로 설치되는 Python 3도 **3.6**입니다.
vllm-omni는 Python 3.10 이상을 요구하므로, 호스트에 직접 설치하려면 여러 장벽이 존재합니다.

| 문제 | 내용 |
|------|------|
| **Python 버전** | 시스템 Python 2.7 / 3.6은 vllm-omni 최소 요건(3.10) 미달 |
| **glibc 버전** | Ubuntu 18.04의 glibc는 2.27. 최신 vllm wheel은 glibc 2.28+ 요구. pip 설치 시 `GLIBC_2.28 not found` 오류 발생 가능 |
| **시스템 라이브러리** | Ubuntu 18.04 기반 네이티브 빌드 환경은 vllm의 C++ 확장(FlashInfer, FA3) 컴파일에 필요한 라이브러리 버전이 낮아 빌드 실패 위험 |
| **venv 우회 시도 한계** | 서버 내 다른 프로젝트(faster-qwen3-tts)의 Python 3.10 venv를 재활용해 새 venv를 만들 수는 있으나, glibc 제약은 해결되지 않음 |

**Docker를 선택한 이유:** `vllm/vllm-openai:v0.18.0` 베이스 이미지는 Python 3.12, glibc 최신, FlashInfer·FA3 등 가속 라이브러리가 모두 포함된 Ubuntu 22.04+ 기반입니다. 호스트 OS 버전과 무관하게 완전한 실행 환경을 제공하며, NVIDIA Container Toolkit을 통해 RTX 3090 GPU를 컨테이너 안에서 그대로 사용할 수 있습니다.

---

## 1. 사전 요구사항

서버에 Docker와 NVIDIA Container Toolkit이 설치되어 있어야 합니다.

```bash
# Docker 설치 확인
docker --version

# NVIDIA runtime 확인
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## 2. 소스 배포

```bash
# 최초 clone
git clone https://github.com/devyoongu/vllm-omni.git ~/tts/vllm-omni

# 이후 업데이트
cd ~/tts/vllm-omni
git pull
```

---

## 3. 모델 다운로드

컨테이너를 이용해 Python 버전 무관하게 다운로드합니다.

```bash
docker run --rm \
  -v /home/posicube/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  python:3.11-slim \
  bash -c "pip install -q huggingface_hub && \
    python -c \"from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')\""
```

> **주의:** 최초 실행 시 수 GB 다운로드가 발생합니다. 이후 실행은 캐시에서 로드합니다.

---

## 4. Docker 이미지 빌드

프로젝트 루트에서 실행합니다.

```bash
cd ~/tts/vllm-omni
docker build -f docker/Dockerfile -t vllm-omni:custom .
```

빌드는 약 5~10분 소요됩니다. 소스가 변경된 경우에만 재빌드가 필요합니다.

---

## 5. 서비스 시작

```bash
cd ~/tts/vllm-omni
docker compose -f docker/docker-compose.yml up -d
```

> **주의:** 명령은 `docker compose` (하이픈 없음, Compose v2 플러그인) 입니다.
> 구버전 `docker-compose` (v1, Python 패키지)는 최신 `requests`/`urllib3` 와 호환되지 않아
> `Not supported URL scheme http+docker` 오류가 발생합니다.

### 로그 확인

```bash
docker logs -f vllm-qwen3-tts
```

### 정상 기동 확인 메시지

```
[AsyncOmniEngine] Stage 0 engine startup completed
[AsyncOmniEngine] Stage 1 engine startup completed
Uvicorn running on http://0.0.0.0:30000
```

### VLLM::Worker 프로세스가 2개인 이유

`nvidia-smi`로 확인하면 `VLLM::Worker` 프로세스가 2개 보입니다. 이는 정상입니다.

```
PID USER DEV     TYPE  GPU        GPU MEM  Command
5658 root   0  Compute  79%   8112MiB  VLLM::Worker  ← Stage 1: Code2Wav
5527 root   0  Compute  19%   8076MiB  VLLM::Worker  ← Stage 0: Thinker
```

Qwen3-TTS는 **2-Stage 아키텍처**로 구성되어 있으며, `--omni` 플래그를 사용하면 vllm-omni의 `Orchestrator`가 각 스테이지를 별도 프로세스로 기동합니다.

| 프로세스 | 스테이지 | 역할 | VRAM |
|---------|---------|------|------|
| Worker #1 | Stage 0: Thinker | 텍스트 → 코덱 토큰 (AR) | ~8 GB |
| Worker #2 | Stage 1: Code2Wav | 코덱 토큰 → PCM 오디오 | ~8 GB |

두 Worker의 VRAM 합산 ~16.2 GB로 RTX 3090 (24 GB) 범위 내에서 정상 운영됩니다.

### Health Check

```bash
curl http://localhost:30000/health
```

---

## 6. 레퍼런스 Voice 등록

서비스 기동 후 TTS에 사용할 레퍼런스 음성을 등록합니다.
`wav/voices.json`에 각 voice의 파일 경로와 발화 텍스트가 정의되어 있습니다.

### 등록

```bash
# 프로젝트 루트(~/tts/vllm-omni)에서 실행
curl -X POST http://localhost:30000/v1/audio/voices \
  -F "name=femail_achernar" \
  -F "consent=agreed" \
  -F "ref_text=안녕하세요 메타엠 시연용 콜봇입니다 무엇을 도와드릴까요" \
  -F "audio_sample=@wav/femail_achernar.wav"

curl -X POST http://localhost:30000/v1/audio/voices \
  -F "name=mail_achird" \
  -F "consent=agreed" \
  -F "ref_text=안녕하세요 메타엠 시연용 콜봇입니다 무엇을 도와드릴까요" \
  -F "audio_sample=@wav/mail_achird.wav"
```

> **중요:** `ref_text`는 해당 wav 파일에 실제로 담긴 발화 내용과 정확히 일치해야 합니다.
> 불일치 시 생성 품질이 크게 저하됩니다 (앞부분에 이상한 소리 발생).

### 등록 목록 확인

```bash
curl http://localhost:30000/v1/audio/voices | python3 -m json.tool
```

### 잘못 등록된 voice 재등록

```bash
# 삭제 후 재등록
curl -X DELETE http://localhost:30000/v1/audio/voices/femail_achernar

curl -X POST http://localhost:30000/v1/audio/voices \
  -F "name=femail_achernar" \
  -F "consent=agreed" \
  -F "ref_text=안녕하세요 메타엠 시연용 콜봇입니다 무엇을 도와드릴까요" \
  -F "audio_sample=@wav/femail_achernar.wav"
```

> **주의:** 컨테이너 재시작 시 등록된 voice가 초기화됩니다. 재시작 후 반드시 재등록이 필요합니다.

---

## 7. 외부 머신에서 접근 (SSH 터널)

배포된 TTS 서비스의 30000 포트는 사내망 방화벽 정책으로 외부에서 직접 접근이 차단될 수 있습니다.
(SSH 22번은 열려있어도 30000은 막힌 케이스가 일반적)

증상 — 클라이언트(Mac/노트북)에서:
- `ping 172.31.88.110` ✅ (L3 도달은 됨)
- `nc -zv 172.31.88.110 30000` ❌ Connection timed out
- `curl http://172.31.88.110:30000/health` ❌ 타임아웃

반면 서버 안에서는 `curl http://localhost:30000/health` 가 정상 200 OK 를 돌려줍니다.

### SSH 로컬 포워딩으로 우회

클라이언트 머신에서 별도 터미널에 띄워두고 유지합니다:

```bash
ssh -i /path/to/ssh_key \
    -N -L 30000:localhost:30000 \
    yh.ahn@172.31.88.110
```

| 옵션 | 의미 |
|------|------|
| `-N` | 원격 명령 실행 없이 터널만 유지 |
| `-L 30000:localhost:30000` | Mac의 30000 → 서버 내부의 localhost:30000 으로 포워딩 |

이후 Mac에서는 `http://localhost:30000` 으로 호출하면 됩니다.

```bash
curl http://localhost:30000/health                   # 200 OK
curl http://localhost:30000/v1/audio/voices          # 등록된 voice 조회
python tests/test_streaming_kdnavien.py              # 스트리밍 TTS 테스트
```

> **참고:** 테스트 스크립트는 `TTS_SERVER` 환경변수로 서버 주소를 오버라이드할 수 있게 되어 있습니다.
> 기본값은 `http://localhost:30000` 이므로 터널만 띄우면 별도 설정 없이 동작합니다.

### 근본 해결 (방화벽 룰 추가)

자주 쓰는 환경이라면 서버 측에 30000 포트 인바운드 룰을 추가합니다 (sudo 필요):

```bash
sudo ufw allow 30000/tcp
# 또는 ufw 미사용 시
sudo iptables -A INPUT -p tcp --dport 30000 -j ACCEPT
```

사내망/클라우드 보안그룹 단에서 막힌 경우엔 위 명령으로 안 뚫리므로 IT 팀에 룰 추가를 요청해야 합니다.

---

## 8. 서비스 중지 / 재시작

```bash
# 중지
docker compose -f docker/docker-compose.yml down

# 재시작 (이미지 재빌드 없이)
docker compose -f docker/docker-compose.yml up -d
```

---

## 9. 소스 업데이트 후 재배포

```bash
cd ~/tts/vllm-omni
git pull

# 소스가 변경된 경우 이미지 재빌드 필요
docker compose -f docker/docker-compose.yml down
docker build -f docker/Dockerfile -t vllm-omni:custom .
docker compose -f docker/docker-compose.yml up -d

# voice 재등록 (컨테이너 재시작으로 초기화됨)
curl -X POST http://localhost:30000/v1/audio/voices \
  -F "name=femail_achernar" \
  -F "consent=agreed" \
  -F "ref_text=안녕하세요 메타엠 시연용 콜봇입니다 무엇을 도와드릴까요" \
  -F "audio_sample=@wav/femail_achernar.wav"
```

---

## 10. 트러블슈팅

### max_model_len 오류

```
User-specified max_model_len (65536) is greater than the derived max_model_len (32768)
```

**원인:** Code2Wav 스테이지가 코덱 스트리밍을 위해 모델의 `max_position_embeddings`를 초과해야 함
**해결:** `docker-compose.yml`에 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 환경변수 설정 (이미 적용됨)

### 음성이 뒤죽박죽으로 들림

**원인:** `ref_text`가 wav 파일의 실제 발화 내용과 불일치
**해결:** voice 삭제 후 올바른 `ref_text`로 재등록 (6번 항목 참고)

### 컨테이너 GPU 인식 불가

```bash
# nvidia-container-toolkit 설치 확인
nvidia-container-cli info
```
