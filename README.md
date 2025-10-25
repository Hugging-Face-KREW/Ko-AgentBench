# Ko-AgentBench
<div align="center">
<img src="https://github.com/user-attachments/assets/9cde519b-7935-4e0f-bd34-4d8a81e14103" width="200">
</div>

## 빠른 시작
```bash
# uv 설치를 먼저 해주세요.

# Python 3.10 + 의존성 설치
uv python install 3.10
uv sync --python 3.10
# (.venv 가상환경이 자동 생성됩니다)
# (선택) 수동 생성: uv venv .venv --python 3.10

# 실행
uv run python your_script.py
```

## 벤치마크 실행 방법

### 기본 사용법

`run_benchmark_with_logging.py`를 사용하여 Ko-AgentBench 데이터셋으로 모델을 평가할 수 있습니다.

```bash
# 기본 실행 (모든 레벨, 캐시 읽기 모드)
uv run python run_benchmark_with_logging.py

# 특정 레벨만 실행
uv run python run_benchmark_with_logging.py --levels L1,L2,L3

# 특정 모델 지정
uv run python run_benchmark_with_logging.py --model openai/gpt-4

# 로컬 모델 사용 (Transformers)
uv run python run_benchmark_with_logging.py --use-local --model Qwen/Qwen2.5-7B-Instruct

# API 호출 및 캐시 저장
uv run python run_benchmark_with_logging.py --cache-mode write
```

### 주요 인자 설명

#### 데이터셋 선택
- `--levels`: 실행할 레벨 지정 (예: `L1,L2,L6,L7`). 기본값은 모든 레벨 실행

#### 모델 설정
- `--model`: 사용할 모델 ID (예: `openai/gpt-4`, `anthropic/claude-3-5-sonnet-20241022`, `gemini/gemini-2.0-flash-exp`)
- `--use-local`: 로컬 Transformers 모델 사용 (API 대신)
- `--quantization`: 로컬 모델 양자화 (`4bit` 또는 `8bit`)
- `--device`: 로컬 추론 장치 (`cuda`, `cpu`, `auto`)
- `--dtype`: Torch dtype (`auto`, `float16`, `bfloat16`, `float32`)

#### 실행 제어
- `--max-steps`: 태스크당 최대 스텝 수 (기본값: 10)
- `--timeout`: 태스크당 타임아웃(초) (기본값: 60)
- `--no-save-logs`: 결과 로그를 저장하지 않음

#### 캐시 모드
- `--cache-mode`: 캐시 동작 방식 (`read` 또는 `write`)
  - `read`: 캐시된 응답만 사용 (기본값, 실제 API 호출 없음)
  - `write`: 실제 API를 호출하고 결과를 캐시에 저장

### 사용 예시

```bash
# 1. GPT-4로 L1-L3 레벨 평가 (캐시 읽기)
uv run python run_benchmark_with_logging.py \
  --levels L1,L2,L3 \
  --model openai/gpt-4

# 2. Claude로 전체 레벨 평가 및 캐시 생성
uv run python run_benchmark_with_logging.py \
  --model anthropic/claude-3-5-sonnet-20241022 \
  --cache-mode write

# 3. 로컬 모델(4bit 양자화)로 평가
uv run python run_benchmark_with_logging.py \
  --use-local \
  --model Qwen/Qwen2.5-7B-Instruct \
  --quantization 4bit \
  --device cuda

# 4. 멀티턴 대화 레벨(L6, L7) 평가
uv run python run_benchmark_with_logging.py \
  --levels L6,L7 \
  --max-steps 20
```

### 결과 확인

실행 결과는 `logs/benchmark_results/by_model/{모델명}/{타임스탬프}/` 경로에 저장됩니다.

각 레벨별로 다음 정보가 포함된 JSON 파일이 생성됩니다:
- 전체 통계 (성공률, 평균 실행시간, 토큰 사용량 등)
- 도구 사용 통계
- 개별 태스크 결과 (tool call, 대화 로그 포함)

## 캐시 시스템

Ko-AgentBench는 API 호출 결과를 캐싱하여 비용 절감과 재현성을 제공합니다.

### 캐시 모드

#### Read 모드 (기본값)
- 캐시된 응답만 사용하며, 실제 API를 호출하지 않습니다
- 캐시 미스 시 에러를 발생시킵니다
- 비용 없이 이전 실험을 재현하거나 분석할 때 사용

```bash
uv run python run_benchmark_with_logging.py --cache-mode read
```

#### Write 모드
- 실제 API를 호출하고 결과를 캐시에 저장합니다
- 새로운 모델을 평가하거나 캐시를 갱신할 때 사용
- write모드를 사용하기 위해서는 `configs/secrets`에 대응하는 APi키를 입력해주세요.

```bash
uv run python run_benchmark_with_logging.py --cache-mode write
```

### 캐시 위치

- 캐시 디렉토리: `bench/cache/`


### 캐시 구조

캐시는 API 요청의 해시를 키로 사용하여 응답을 저장합니다. 이를 통해:
- 동일한 요청에 대해 일관된 응답 보장
- API 비용 절감
- 실험 재현성 향상
- 오프라인 평가 가능
