#!/bin/bash
# 서버 기동 후 voices.json에 정의된 음성 샘플을 자동 등록하는 스크립트
# docker-compose entrypoint에서 vllm 서버를 백그라운드로 띄운 뒤 호출

set -e

VOICES_JSON="/app/vllm-omni/wav/voices.json"
WAV_DIR="/app/vllm-omni/wav"
SERVER_URL="http://localhost:30000"
MAX_WAIT=300  # 최대 5분 대기

if [ ! -f "$VOICES_JSON" ]; then
    echo "[register_voices] voices.json not found at $VOICES_JSON, skipping."
    exit 0
fi

# 서버가 준비될 때까지 대기
echo "[register_voices] Waiting for server to be ready..."
elapsed=0
until curl -sf "$SERVER_URL/health" > /dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [ $elapsed -ge $MAX_WAIT ]; then
        echo "[register_voices] ERROR: Server not ready after ${MAX_WAIT}s, giving up."
        exit 1
    fi
done
echo "[register_voices] Server is ready (waited ${elapsed}s)."

# 이미 등록된 음성 목록 조회
existing=$(curl -sf "$SERVER_URL/v1/audio/voices" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for v in data.get('uploaded_voices', []):
    print(v['name'])
" 2>/dev/null || true)

# voices.json 파싱 후 등록
python3 -c "
import json, subprocess, sys, os

with open('$VOICES_JSON') as f:
    voices = json.load(f)

existing = set('''$existing'''.strip().split('\n')) - {''}
wav_dir = '$WAV_DIR'
server_url = '$SERVER_URL'

for name, info in voices.items():
    if name in existing:
        print(f'[register_voices] {name} already registered, skipping.')
        continue

    # wav 파일은 컨테이너 내부의 wav 디렉토리에서 찾기
    wav_file = os.path.join(wav_dir, os.path.basename(info['ref_audio']))
    if not os.path.exists(wav_file):
        print(f'[register_voices] WARNING: {wav_file} not found, skipping {name}.')
        continue

    ref_text = info.get('ref_text', '')
    cmd = [
        'curl', '-sf', '-X', 'POST', f'{server_url}/v1/audio/voices',
        '-F', f'audio_sample=@{wav_file}',
        '-F', f'name={name}',
        '-F', 'consent=agreed',
        '-F', f'ref_text={ref_text}',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f'[register_voices] Registered: {name}')
    else:
        print(f'[register_voices] FAILED to register {name}: {result.stderr}', file=sys.stderr)
"

echo "[register_voices] Done."
