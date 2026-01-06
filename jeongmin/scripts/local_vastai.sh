#!/bin/bash
set -e

# ============================================================
# 로컬에서 Vast.ai 인스턴스 실행 스크립트
# (vastai 기본 이미지 사용, GitHub에서 프로젝트 clone)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 사용법 출력
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Vast.ai에서 학습을 실행합니다. (vastai 기본 이미지 사용)

Options:
    --dry-run                   Search only, don't create instance
    --search                    Search available instances only
    -h, --help                  Show this help message

Environment Variables (Required):
    VAST_API_KEY                Vast.ai API key
    HF_TOKEN                    HuggingFace token
    GH_TOKEN                    GitHub token (for private repo clone)
    GH_USER                     GitHub username
    GH_REPO                     GitHub repository name

Environment Variables (Optional):
    GH_BRANCH                   Git branch (default: main)
    WANDB_API_KEY               Weights & Biases API key
    AWS_ACCESS_KEY_ID           AWS access key
    AWS_SECRET_ACCESS_KEY       AWS secret key
    AWS_DEFAULT_REGION          AWS region (default: ap-northeast-2)
    S3_DATA_PATH                S3 data path

Examples:
    # 기본 실행
    ./scripts/local_vastai.sh

    # Dry run (인스턴스 생성 안함)
    ./scripts/local_vastai.sh --dry-run

    # 검색만
    ./scripts/local_vastai.sh --search
EOF
    exit 0
}

# 환경변수 확인
check_env() {
    local missing=()

    if [ -z "$VAST_API_KEY" ]; then
        missing+=("VAST_API_KEY")
    fi

    if [ -z "$HF_TOKEN" ]; then
        missing+=("HF_TOKEN")
    fi

    if [ -z "$GH_TOKEN" ]; then
        missing+=("GH_TOKEN")
    fi

    if [ -z "$GH_USER" ]; then
        missing+=("GH_USER")
    fi

    if [ -z "$GH_REPO" ]; then
        missing+=("GH_REPO")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required environment variables:"
        for var in "${missing[@]}"; do
            echo "  - $var"
        done
        echo ""
        echo "Set them in your shell or create a .env file:"
        echo "  export VAST_API_KEY=your_vastai_api_key"
        echo "  export HF_TOKEN=your_huggingface_token"
        echo "  export GH_TOKEN=your_github_token"
        echo "  export GH_USER=your_github_username"
        echo "  export GH_REPO=your_repo_name"
        exit 1
    fi
}

# 인자 파싱
DRY_RUN=""
SEARCH=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --search)
            SEARCH="--search"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# 메인 실행
main() {
    log_info "Starting Vast.ai launch pipeline"
    log_info "Project directory: $PROJECT_DIR"
    echo ""

    check_env

    cd "$PROJECT_DIR"

    log_info "Launching on Vast.ai..."
    log_info "GitHub: ${GH_USER}/${GH_REPO} (branch: ${GH_BRANCH:-main})"
    echo ""

    python3 scripts/vastai_launcher.py \
        --wandb-key "${WANDB_API_KEY:-}" \
        --hf-token "${HF_TOKEN:-}" \
        --aws-key "${AWS_ACCESS_KEY_ID:-}" \
        --aws-secret "${AWS_SECRET_ACCESS_KEY:-}" \
        --aws-region "${AWS_DEFAULT_REGION:-ap-northeast-2}" \
        --s3-path "${S3_DATA_PATH:-}" \
        --gh-token "${GH_TOKEN:-}" \
        --gh-user "${GH_USER:-}" \
        --gh-repo "${GH_REPO:-}" \
        --gh-branch "${GH_BRANCH:-main}" \
        $DRY_RUN \
        $SEARCH \
        $EXTRA_ARGS

    echo ""
    log_info "Done!"
}

main
