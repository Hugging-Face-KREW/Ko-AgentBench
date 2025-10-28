# Ko-AgentBench

[English](README_en.md) | 한국어

<div align="center">
<img src="https://github.com/user-attachments/assets/9cde519b-7935-4e0f-bd34-4d8a81e14103" width="200">
</div>

Ko-AgentBench는 한국어 도구 사용(Tool-Calling) 에이전트를 평가하기 위한 벤치마크입니다.  
실제 API를 직접 호출하지 않고도 테스트 가능한 캐시 기반 가상 API를 제공하여 재현성, 일관성, 비용 안정성을 확보합니다.

---

## Ko-AgentBench ✨

도구 호출 에이전트를 평가하는 벤치마크는 단순히 "정확한 API를 호출했는가"를 넘어서야 합니다. 에이전트는 불완전한 정보 속에서 다수의 도구 중 적절한 것을 선택하고, 때로는 여러 단계에 걸쳐 도구를 연결하며, 오류가 발생했을 때 적절히 대응해야 합니다. 또한 동일한 정보를 반복 요청하지 않고 효율적으로 작동해야 하며, 여러 턴에 걸친 대화에서 맥락을 유지해야 합니다.

Ko-AgentBench는 이러한 문제의식에서 출발하여, 에이전트의 도구 호출 능력을 현실성(Realism), 명확성(Clarity), 판별력(Discriminative Power), 견고성(Robustness), 효율성(Efficiency), 재현성(Reproducibility), 확장성(Extensibility)의 원칙을 기반으로 평가합니다.

현실성을 위해 단일 API 호출로 완결되는 단편적 태스크가 아닌, 실제 업무 흐름에서 발생하는 stateful 다중 턴 시나리오를 구성했습니다. 도구 간 데이터 의존성과 실행 흐름이 실제 환경과 유사하게 설계되어, 에이전트가 실제 환경에서 직면할 문제를 반영합니다. 명확성을 확보하기 위해 평가 대상 태스크, 입출력 형식, 측정 지표, 스키마 정의, 평가 절차를 모호함 없이 정의했습니다.

판별력은 반복 실행을 통한 Pass@k 메트릭으로 달성됩니다. 동일 태스크를 k번 실행하여 성공률을 측정함으로써 모델 간 성능 차이를 통계적으로 유의미하게 구분할 수 있습니다. 견고성 평가를 위해 오류 주입 시나리오와 불완전한 정보 상황을 의도적으로 설계하여, 에이전트가 실패 시 안전하게 처리하는 능력(graceful degradation)을 측정합니다. 효율성은 태스크 완수 여부뿐만 아니라, 최소 단계 실행, 중복 호출 억제, 이전 결과 재사용 등을 정량화하여 평가합니다.

재현성 확보를 위해 가상 API 시스템과 오프라인 실행 모드를 제공합니다. 네트워크 상태나 외부 API 변동성과 무관하게 동일한 결과를 재현할 수 있으며, 시드 및 파라미터 고정을 통해 실험 재현성을 보장합니다. 이는 연구자들이 공정한 조건에서 모델을 비교할 수 있게 하며, 벤치마크 결과의 신뢰성을 높입니다. 확장성은 태스크, 도구, 평가 지표의 모듈화를 통해 구현되어, 새로운 평가 시나리오를 손쉽게 추가할 수 있습니다.

### 7가지 독립적 Task 실행를 통한 모델의 성능 측정

에이전트의 도구 호출 능력은 단일 차원으로 측정할 수 없습니다. "도구를 잘 쓴다"는 것은 정확한 도구를 선택하는 능력, 여러 도구를 연결하는 계획 능력, 오류에 대응하는 강건성, 효율적으로 작동하는 능력 등 여러 독립적인 역량의 조합입니다. Ko-AgentBench는 이러한 역량을 7가지로 분리하여 각각 독립적으로 측정합니다. 이는 난이도 단계가 아니라, 서로 다른 측면의 능력을 평가하는 체계입니다.

**L1(단일 도구 실행)**은 가장 기본적인 능력으로, 주어진 도구를 올바른 파라미터로 실행할 수 있는지를 검증합니다. 이는 API 스키마 이해도와 파라미터 매핑 정확성을 측정합니다. **L2(도구 선택)**는 다수의 후보 도구 중 자연어 요청에 가장 적합한 API를 선택하는 능력을 평가합니다. 이는 현실에서 에이전트가 직면할 수 있는 가장 빈번한 상황으로, 의도 파악과 도구 매핑 능력을 측정합니다.

**L3(순차적 추론)**과 **L4(병렬적 추론)**는 복수 도구 사용에서 요구되는 서로 다른 추론 패턴을 평가합니다. L3는 도구 체이닝(Chaining) 능력으로, 한 도구의 출력을 다른 도구의 입력으로 연결하는 계획 수립 및 실행을 측정합니다. 예를 들어 "11번가에서 주문 조회 → 배송 번호 추출 → 택배사 API로 배송 추적"과 같은 순차적 파이프라인을 구성할 수 있는지 평가합니다. L4는 병렬 호출 후 결과 종합(Aggregation) 능력으로, 독립적인 여러 도구를 동시에 호출하고 결과를 비교·분석하여 결론을 도출하는 능력을 측정합니다.

**L5(강건성)**는 오류 상황 대처 능력을 평가합니다. 정보가 부족할 때 추가 질문을 요청하는지, API 호출이 실패했을 때 존재하지 않는 결과를 만들어내지 않는지(Hallucination 방지), 대체 경로(Fallback)를 활용할 수 있는지를 측정합니다. 이는 현실 환경에서 에이전트의 신뢰성을 결정하는 핵심 요소입니다.

**L6(효율성)**은 멀티턴 대화에서 이전 도구 호출 결과를 효율적으로 재활용하는 능력을 평가합니다. "쿠팡과 네이버 가격 비교해줘" 후 "네이버 가격 얼마였지?"라는 질문에 불필요하게 API를 재호출하지 않고 이전 결과를 활용할 수 있는지를 측정합니다. 이는 비용과 지연을 최적화하는 능력입니다.

**L7(장기 컨텍스트 기억)**은 여러 턴에 걸쳐 맥락을 유지하고 이전 정보를 활용하는 능력을 평가합니다. "제주도 여행 갈 거야" 후 몇 턴 뒤 "날씨 어때?"라는 질문에 제주도 맥락을 유지하여 올바른 API를 호출할 수 있는지를 측정합니다.

이러한 7가지 역량은 각각 독립적으로 측정되며, 종합 점수는 이들의 가중 평균으로 산출됩니다. 이를 통해 모델의 강점과 약점을 세밀하게 파악할 수 있으며, 개선이 필요한 영역을 명확히 식별할 수 있습니다.

---

## 🚀 빠른 시작

### 1) 설치
```bash
# 저장소 클론
git clone https://github.com/Hugging-Face-KREW/Ko-AgentBench
cd Ko-AgentBench

# Python 3.10 + 의존성 설치 (uv 사용)
uv python install 3.10
uv sync --python 3.10
```

### 2) 환경 변수 설정 (.env)
```bash
# 모델별 API 키 (필요한 항목만)
OPENAI_API_KEY=your_key
AZURE_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GEMINI_API_KEY=your_key

# 로컬 모델용 OpenAI 호환 서버
OPENAI_BASE_URL=http://localhost:8000/v1

# 재현성 설정
KO_AGENTBENCH_OFFLINE=0        # 1이면 캐시만 사용
KO_AGENTBENCH_SEED=2025
```

### 3) 실행과 평가
```bash
# 벤치마크 실행 (L1 레벨, 캐시 읽기 모드)
uv run python run_benchmark_with_logging.py --levels L1 --model openai/gpt-4

# 평가 (실행 날짜를 YYYYMMDD 형식으로 입력)
uv run python evaluate_model_run.py --date 20251022 --model openai/gpt-4 --format all
```

---

## 🧪 파이프라인 구조

```
[1] 실행: run_benchmark_with_logging.py
    └─ 실행 로그 → logs/ 디렉토리
         (대화, 도구 호출, 파라미터, 캐시 적중률, 오류)

[2] 평가: evaluate_model_run.py
    └─ 평가 보고서 → reports/{model}_{date}/
         (지표 집계, JSON/CSV/Markdown 형식)
```

실행과 평가를 분리하여 독립적인 재현과 자동화가 가능합니다.

---

## ⚡ 벤치마크 실행

### 기본 사용법
run_benchmark_with_logging.py로 모델을 평가합니다.

```bash
# 전체 레벨 실행 (캐시 읽기 모드)
uv run python run_benchmark_with_logging.py

# 특정 레벨만 실행
uv run python run_benchmark_with_logging.py --levels L1,L2,L3

# 특정 모델 지정
uv run python run_benchmark_with_logging.py --model openai/gpt-4

# 로컬 모델 사용
uv run python run_benchmark_with_logging.py --use-local --model Qwen/Qwen2.5-7B-Instruct

# API 호출 후 캐시 저장
uv run python run_benchmark_with_logging.py --cache-mode write
```

### 주요 옵션
**데이터셋 선택**
- `--levels`: 실행할 레벨 (예: `L1,L2,L6,L7`) — 기본값: 전체

**모델 설정**
- `--model`: 모델 ID (예: `openai/gpt-4`, `anthropic/claude-3-5-sonnet-20241022`)
- `--use-local`: 로컬 Transformers 사용
- `--quantization`: `4bit`/`8bit`
- `--device`: `cuda`/`cpu`/`auto`
- `--dtype`: `auto`/`float16`/`bfloat16`/`float32`

**실행 제어**
- `--max-steps`: 태스크당 최대 단계 (기본: 10)
- `--timeout`: 태스크당 시간 제한(초) (기본: 60)
- `--passes`: 반복 실행 횟수 (Pass@k 계산용)
- `--no-save-logs`: 로그 저장 비활성화

**캐시 모드**
- `--cache-mode`:
    - `read` (기본): 저장된 캐시만 사용
    - `write`: 실제 API 호출 후 캐시 저장 (configs/secrets에 API 키 필요)

**결과 저장 위치**
- 기본: `logs/benchmark_results/by_model/{모델명}/{타임스탬프}/`

---

## 📏 평가

### 사용법
evaluate_model_run.py로 실행 로그를 분석하여 보고서를 생성합니다.

```bash
# 기본 평가
python evaluate_model_run.py --date 20251022 --model azure/gpt-4o

# 빠른 테스트 (레벨당 1개)
python evaluate_model_run.py --date 20251022 --model azure/gpt-4o --quick
```

### 주요 옵션
- `--date`: 벤치마크 실행 날짜 (YYYYMMDD 형식)
- `--model`: 평가할 모델 ID
- `--judge-models`: 평가 모델 목록 (기본: `gpt-4o,claude-sonnet,gemini`)
- `--sample N`: 레벨당 N개만 평가
- `--quick`: 레벨당 1개만 평가 (샘플링)
- `--format`: 출력 형식 (`json`/`csv`/`markdown`/`all`)

### 결과 위치
- 출력: `reports/{model}_{date}/`
    - `evaluation_report.json`
    - `evaluation_summary.csv`
    - `evaluation_report.md`

---

## 🔄 캐시 시스템과 재현성

**동작 원리**
- API 요청과 응답을 저장하여 동일 요청 시 캐시된 응답 반환
- 캐시 키: `hash(method, url, sorted(query), sorted(headers), body)`
- 오프라인 모드: `KO_AGENTBENCH_OFFLINE=1` 설정 시 네트워크 차단
- 캐시 적중률을 로그와 보고서에 자동 기록

**캐시 모드**
- **Read** (기본): 캐시만 사용, 없으면 오류. 재현과 분석에 최적
- **Write**: 실제 API 호출 후 응답 저장. 초기 생성이나 갱신 시 사용

**저장 위치**
- 디렉토리: `bench/cache/`
- 내용: 요청 해시별 응답 본문, 헤더, 메타데이터

---

## 🧩 평가 레벨과 태스크

에이전트의 도구 호출 능력을 7개 레벨로 평가합니다.

| Level | 평가 영역 | 예시 | 주요 지표 |
|-------|----------|------|-----------|
| **L1** | 단일 도구 호출 | `StockPrice_ls(symbol="005930")` | ToolAcc, ArgAcc, CallEM, RespOK |
| **L2** | 도구 선택 | `get_crypto_price_upbit` vs `get_crypto_price_bithumb` | SelectAcc |
| **L3** | 순차 실행 | 종목검색 → 시세조회 → 결과종합 | FSM, PSM, ProvAcc |
| **L4** | 병렬 처리 | 여러 거래소 동시 조회 후 비교 | Coverage, SourceEPR |
| **L5** | 오류 처리 | API 실패 시 대체 경로 활용 | ErrorDetect, FallbackSR |
| **L6** | 효율성 | 중복 호출 방지, 캐시 활용 | EffScore, ReuseRate |
| **L7** | 맥락 유지 | 멀티턴 대화에서 이전 정보 활용 | ContextRetention, RefRecall |

---

## 🧩 평가 지표

| 레벨 | 지표 | 설명 | 계산 방식 |
|---|---|---|---|
| 공통 | **RRR** | 정상 실행 비율 | 성공 실행 수 / 전체 실행 수 |
| 공통 | **SR** | 태스크 완수 점수 | Judge 평가 1~5점 → (점수-1)/4 |
| 공통 | **EPRCVR** | 유효 도구 호출 비율 | 유효 호출 수 / 전체 호출 수 |
| 공통 | **PassAtK** | k회 시도 성공률 | 성공 시도 수 / k |
| L1 | **ToolAcc** | 올바른 도구 선택 | 일치=1, 불일치=0 |
| L1 | **ArgAcc** | 인자 정확도 | Judge 평가 1~5 → 0~1 |
| L1 | **CallEM** | 도구+인자 완전 일치 | 0 또는 1 |
| L1 | **RespOK** | 응답 형식 준수 | 0 또는 1 |
| L2 | **SelectAcc** | 올바른 도구 선택률 | 0 또는 1 |
| L3 | **FSM** | 호출 순서 일치 | 0 또는 1 |
| L3 | **PSM** | 필수 도구 포함률 | 포함된 필수 도구 / 전체 필수 도구 |
| L3 | **DeltaStepsNorm** | 효율성 | min(1, 최소 단계 / 실제 단계) |
| L3 | **ProvAcc** | 인자 전달 정확도 | 비율 |
| L4 | **Coverage** | 필수 도구 실행률 | 성공한 필수 도구 / 전체 필수 도구 |
| L4 | **SourceEPR** | 도구별 유효 호출률 | 평균(유효 호출 / 전체 호출) |
| L5 | **ErrorDetect** | 오류 감지율 | 0~1 정규화 |
| L5 | **GracefulFail** | 안전한 실패율 | 안전 실패 / 전체 실패 |
| L5 | **FallbackSR** | 대체 경로 성공률 | 비율 |
| L6 | **ReuseRate** | 재사용률 | 재사용 / (재사용+중복) |
| L6 | **RedundantCallRate** | 중복 호출률 | 중복 호출 / 전체 호출 |
| L6 | **EffScore** | 성공 시 효율 점수 | min(1, 최소 단계 / 실제 단계) |
| L7 | **ContextRetention** | 맥락 유지 능력 | Judge 평가 1~5 → 0~1 |
| L7 | **RefRecall** | 정보 회상 정확도 | Judge 평가 1~5 → 0~1 |

### Judge 평가
- 평가 모델: GPT-4o, Claude, Gemini 앙상블
- 점수 집계: 평균 또는 중앙값
- 모델명을 제거한 블라인드 평가로 공정성 확보

---

## 🧮 종합 점수

- **기본 능력** = L1~L3 지표 평균 (40%)
- **오류 처리** = L5 지표 평균 (20%)
- **효율성** = L6 지표 평균 (25%)
- **맥락 처리** = L7 지표 평균 (15%)

---

## 📊 리더보드

평가 결과는 `reports/{model}_{date}/`에 자동 저장됩니다:
- JSON (`evaluation_report.json`)
- CSV (`evaluation_summary.csv`)
- Markdown (`evaluation_report.md`)

---

## 📁 폴더 구조
```
Ko-AgentBench/
├─ bench/
│  ├─ tasks/       # YAML 태스크 정의
│  ├─ tools/       # 도구 스펙과 어댑터
│  ├─ runner/      # 실행 엔진과 메트릭
│  └─ cache/       # API 응답 캐시
├─ logs/           # 실행 로그
├─ reports/        # 평가 보고서
├─ configs/        # 설정 파일
├─ run_benchmark_with_logging.py
├─ evaluate_model_run.py
└─ README.md
```

---

## 🔍 사용 예시

```bash
# 1) GPT-4로 L1-L3 레벨 평가
uv run python run_benchmark_with_logging.py --levels L1,L2,L3 --model openai/gpt-4

# 2) Claude로 전체 레벨 평가 + 캐시 생성
uv run python run_benchmark_with_logging.py --model anthropic/claude-3-5-sonnet-20241022 --cache-mode write

# 3) 로컬 모델 4bit 양자화 평가
uv run python run_benchmark_with_logging.py --use-local --model Qwen/Qwen2.5-7B-Instruct --quantization 4bit --device cuda

# 4) 멀티턴 대화 레벨 평가
uv run python run_benchmark_with_logging.py --levels L6,L7 --max-steps 20

# 5) 평가 보고서 생성
python evaluate_model_run.py --date 20251022 --model azure/gpt-4o --format all

# 6) 빠른 샘플 평가
python evaluate_model_run.py --date 20251022 --model azure/gpt-4o --quick
```

---

## ⚖️ 라이선스

[Apache-2.0](LICENSE) 라이선스를 따릅니다.