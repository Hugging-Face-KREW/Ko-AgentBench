#!/bin/bash
# Ko-AgentBench 다중 모델 실험 스크립트

set -e

#==============================================================================
# 🔧 재활용 시 수정이 필요한 설정 (CONFIGURATION)
#==============================================================================

# 실험 기본 설정
DEFAULT_MAX_STEPS=10              # 태스크당 최대 스텝 수
DEFAULT_TIMEOUT=60                # 태스크당 타임아웃 (초)
DEFAULT_QUANTIZATION="none"       # 양자화 방법: 4bit | 8bit | none
DEFAULT_DEVICE="auto"             # 디바이스: auto | cuda | cpu
DEFAULT_DTYPE="bfloat16"          # 데이터 타입: auto | float16 | bfloat16 | float32

# 실행 대상 설정 (기본값, CLI 인자로 오버라이드 가능)
DEFAULT_LEVELS=""                 # 비어있으면 전체 레벨 (L1~L7), 예: "L1,L2"
DEFAULT_MODELS=""                 # 비어있으면 레지스트리 전체, 예: "Qwen/Qwen3-4B,skt/A.X-4.0"

#==============================================================================
# 내부 변수 (수정 불필요)
#==============================================================================

# 색상
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

# 런타임 변수
MAX_STEPS=$DEFAULT_MAX_STEPS
TIMEOUT=$DEFAULT_TIMEOUT
QUANTIZATION=$DEFAULT_QUANTIZATION
DEVICE=$DEFAULT_DEVICE
DTYPE=$DEFAULT_DTYPE
LEVELS=$DEFAULT_LEVELS
SPECIFIC_MODELS=$DEFAULT_MODELS
SKIP_FAILED=false
EXPERIMENT_ID=""

# 로깅 함수
log() { echo -e "${BLUE}[$(date '+%H:%M:%S')] $1${NC}"; }
log_success() { echo -e "${GREEN}[OK] $1${NC}"; }
log_error() { echo -e "${RED}[ERR] $1${NC}"; }
log_info() { echo -e "${CYAN}[INFO] $1${NC}"; }

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
  --levels LEVELS       테스트할 레벨 (기본: ${DEFAULT_LEVELS:-all})
                        예: --levels L1,L2,L3
  
  --max-steps N         태스크당 최대 스텝 (기본: $DEFAULT_MAX_STEPS)
  --timeout N           태스크당 타임아웃 초 (기본: $DEFAULT_TIMEOUT)
  --quantization M      양자화: 4bit|8bit|none (기본: $DEFAULT_QUANTIZATION)
  --device D            디바이스: auto|cuda|cpu (기본: $DEFAULT_DEVICE)
  --dtype D             데이터타입: auto|float16|bfloat16 (기본: $DEFAULT_DTYPE)
  
  --models MODELS       특정 모델만 테스트 (기본: 레지스트리 전체)
                        예: --models "Qwen/Qwen3-4B,skt/A.X-4.0"
  
  --skip-failed         실패해도 다음 모델 계속 진행
  --experiment-id ID    커스텀 실험 ID
  --help                도움말 표시

EXAMPLES:
  $0                                    # 전체 모델, 전체 레벨
  $0 --levels L1,L2                    # L1, L2만 실행
  $0 --models "Qwen/Qwen3-4B"          # 특정 모델만
  $0 --skip-failed                     # 실패 무시하고 계속 진행
EOF
}

# 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --levels) LEVELS="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --models) SPECIFIC_MODELS="$2"; shift 2 ;;
        --skip-failed) SKIP_FAILED=true; shift ;;
        --experiment-id) EXPERIMENT_ID="$2"; shift 2 ;;
        --help) print_usage; exit 0 ;;
        *) log_error "Unknown: $1"; print_usage; exit 1 ;;
    esac
done

get_model_list() {
    uv run python -c "import sys; sys.path.insert(0, '.'); from bench.models import MODEL_IDS; print('\n'.join(MODEL_IDS))"
}

run_model_benchmark() {
    local model="$1" start=$(date +%s)
    echo ""; echo "========== MODEL: $model =========="; echo ""
    
    # 모든 레벨을 쉼표로 구분된 하나의 문자열로 만듦
    local levels_str
    if [[ -n "$LEVELS" ]]; then
        levels_str="$LEVELS"
    else
        levels_str="L1,L2,L3,L4,L5,L6,L7"
    fi
    
    echo "--- $model / $levels_str ---"
    
    # 모든 레벨을 한 번에 실행
    local cmd=(uv run python run_benchmark_with_logging.py --model "$model" --levels "$levels_str" --max-steps "$MAX_STEPS" 
               --timeout "$TIMEOUT" --device "$DEVICE" --dtype "$DTYPE" --use-local)
    [[ -n "$QUANTIZATION" && "$QUANTIZATION" != "none" ]] && cmd+=(--quantization "$QUANTIZATION")
    
    local success=true
    if "${cmd[@]}"; then
        local total=$(($(date +%s) - start))
        log_success "$model done: ${total}s"
        echo "$model:success:$total:all_levels" >> "$RESULTS_SUMMARY"
        return 0
    else
        local code=$? total=$(($(date +%s) - start))
        log_error "$model failed: ${total}s (exit code: $code)"
        echo "$model:failed:$total:all_levels:$code" >> "$RESULTS_SUMMARY"
        return 1
    fi
}

save_experiment_summary() {
    local id="$1" dir="logs/experiment_summaries" file="$dir/${id}_summary.txt"
    mkdir -p "$dir"
    
    {
        echo "======================================"
        echo "Ko-AgentBench Experiment Summary"
        echo "======================================"
        echo "Experiment ID: $id"
        echo "Timestamp: $(date -Iseconds)"
        echo ""
        echo "Configuration:"
        echo "  Levels: ${LEVELS:-all}"
        echo "  Max Steps: $MAX_STEPS"
        echo "  Timeout: $TIMEOUT"
        echo "  Quantization: $QUANTIZATION"
        echo "  Device: $DEVICE"
        echo "  Dtype: $DTYPE"
        echo ""
        echo "Results:"
        while IFS=':' read -r model status time _; do
            echo "  $model: $status (${time}s)"
        done < "$RESULTS_SUMMARY"
    } > "$file"
    
    log_success "Summary: $file"
}

main() {
    echo "========== Ko-AgentBench Experiment =========="
    [[ -z "$EXPERIMENT_ID" ]] && EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)"
    log_info "ID: $EXPERIMENT_ID"
    
    RESULTS_SUMMARY="/tmp/ko_bench_${EXPERIMENT_ID}.tmp"; > "$RESULTS_SUMMARY"
    
    local models=(); [[ -n "$SPECIFIC_MODELS" ]] && IFS=',' read -ra models <<< "$SPECIFIC_MODELS" || readarray -t models <<< "$(get_model_list)"
    [[ ${#models[@]} -eq 0 ]] && { log_error "No models!"; exit 1; }
    log_info "Models: ${#models[@]}, Levels: ${LEVELS:-all}"
    
    log "Starting..."
    for i in "${!models[@]}"; do
        echo ""; echo "▶▶▶ Model $((i+1))/${#models[@]}: ${models[$i]} ◀◀◀"
        run_model_benchmark "${models[$i]}" || [[ "$SKIP_FAILED" != "true" ]] && { log_error "Stopped"; break; }
    done
    
    echo ""; echo "========== SUMMARY =========="
    local total=0 success=0 ttime=0
    
    while IFS=':' read -r m s t d _; do
        ((total++)); ((ttime += t)); [[ "$s" == "success" ]] && ((success++))
        echo "  $([[ $s == success ]] && echo -e "${GREEN}✅" || echo -e "${RED}❌")${NC} $m (${t}s)"
    done < "$RESULTS_SUMMARY"
    
    echo ""; log_info "Total: $total | Success: $success | Time: ${ttime}s ($((ttime/60))m)"
    save_experiment_summary "$EXPERIMENT_ID"
    rm -f "$RESULTS_SUMMARY"
    echo "========== Done! Results in logs/ =========="
    [[ $success -gt 0 ]] && exit 0 || exit 1
}

main "$@"