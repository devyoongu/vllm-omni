# 공식 0.18.0 이미지를 사용하여 모든 가속 라이브러리(FA3, FlashInfer) 확보
FROM vllm/vllm-openai:v0.18.0

# 오디오 처리를 위한 필수 시스템 패키지 + 빌드용 git 추가
USER root
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/vllm-omni
COPY . .

# vllm-omni 설치 시 버전 감지 에러를 방지하기 위해 임시 버전 환경변수 설정
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0

# HuggingFace 캐시 경로 지정 (docker-compose 볼륨 마운트 경로와 일치)
ENV HF_HOME=/root/.cache/huggingface

# vLLM-Omni 소스 설치
RUN pip install --no-cache-dir -e .
