#!/bin/bash

echo "=========================================="
echo "Starting training pipeline..."
echo "=========================================="

source /venv/main/bin/activate

# Discord 로그 알림 함수
send_discord_log() {
    local webhook_url="https://discord.com/api/webhooks/1454916374710452265/Ds7Xn1BUv4q2fYEFbbXO5jJMcqdZvp9PzMIY0lNl1jdo7mldh12xc13runxLOKiLxAiB"

    if [ -n "$webhook_url" ]; then
        local log_tail=""

        # vastai logs 명령어로 로그 가져오기
        if [ -n "$CONTAINER_ID" ] && [ -n "$VAST_API_KEY" ]; then
            log_tail=$(vastai logs "$CONTAINER_ID" --api-key "$VAST_API_KEY" 2>&1 | tail -n 30)
            echo "Log 가져오기 성공"
        fi

        if [ -z "$log_tail" ]; then
            log_tail="로그를 가져올 수 없음 (CONTAINER_ID 또는 VAST_API_KEY 누락)"
        fi

        # Discord 2000 character limit - truncate log to ~1800 chars (leaving room for header + formatting)
        log_tail=$(echo "$log_tail" | head -c 1800)

        local message="**Instance Destroyed** (Exit Code: $TRAIN_EXIT_CODE)\n\`\`\`\n${log_tail}\n\`\`\`"
        local escaped=$(echo "$message" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

        curl -s -H "Content-Type: application/json" \
             -X POST \
             -d "{\"content\": ${escaped}}" \
             "$webhook_url"
        echo "Discord notification sent!"
    else
        echo "DISCORD_WEBHOOK_URL not set, skipping notification"
    fi
}

# 인스턴스 삭제 함수
destroy_instance() {
    send_discord_log
    sleep 20

    if [ -n "$CONTAINER_ID" ] && [ -n "$VAST_API_KEY" ]; then
        echo "Destroying Vast.ai instance: $CONTAINER_ID"
        vastai destroy instance "$CONTAINER_ID" --api-key "$VAST_API_KEY"
        echo "Instance destroyed!"
    else
        echo "CONTAINER_ID or VAST_API_KEY not set, skipping instance destruction"
    fi
}

# 1. HuggingFace 로그인 (필수)
if [ -n "$HF_TOKEN" ]; then
    echo "[1/4] Logging into HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"
    echo "HuggingFace login complete!"
else
    echo "[ERROR] HF_TOKEN not set. Cannot proceed without HuggingFace login."
    echo "Destroying instance due to missing HF_TOKEN..."
    destroy_instance
    exit 1
fi

# 2. S3에서 데이터 다운로드
if [ -n "$S3_DATA_PATH" ]; then
    echo "[2/4] Downloading data from S3: $S3_DATA_PATH"
    # mkdir -p /workspace/data
    # aws s3 sync "$S3_DATA_PATH" /workspace/data/ --quiet
    aws s3 cp s3://pcc-777-navy/2025-12-29:19-43-44/checkpoint-3500/ ./output --recursive
    echo "Data download complete!"
else
    echo "[2/4] S3_DATA_PATH not set, skipping data download"
fi

# 3. W&B 로그인 (필수)
if [ -n "$WANDB_API_KEY" ]; then
    echo "[3/4] Logging into W&B..."
    wandb login "$WANDB_API_KEY"
    echo "W&B login complete!"
else
    echo "[WARN] WANDB_API_KEY not set. Continuing without W&B logging."
fi

# 4. 학습 실행
echo "[4/4] Starting training..."
echo "=========================================="
echo "Data download complete!"
######################################################################### 여기서 하드 코딩 #########################################################################
python3 -u src/main.py fit -c configs/config.yaml $@
TRAIN_EXIT_CODE=$?

echo "=========================================="
echo "Training finished with exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

# 5. 학습 완료/실패 후 Vast.ai 인스턴스 종료
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "Destroying instance due to training failure..."
fi
destroy_instance

exit $TRAIN_EXIT_CODE
