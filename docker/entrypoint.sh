#!/bin/bash
# vllm 서버를 기동한 뒤, 음성 샘플을 자동 등록하는 entrypoint
set -e

# 1) vllm serve를 백그라운드로 시작 (CMD에서 전달받은 인자 사용)
echo "[entrypoint] Starting vllm serve with args: $@"
vllm serve "$@" &
VLLM_PID=$!

# 2) 음성 샘플 자동 등록 (백그라운드)
/app/vllm-omni/docker/register_voices.sh &

# 3) vllm 프로세스를 foreground로 유지
wait $VLLM_PID
