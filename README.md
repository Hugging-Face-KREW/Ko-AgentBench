# Ko-AgentBench
<img width="1200" height="800" alt="banner" src="https://github.com/user-attachments/assets/b65de588-e5ff-4d4b-a386-0e1d0c684cf4" />

[English](README_en.md) | 한국어

<div align="center">
<img src="https://github.com/user-attachments/assets/9cde519b-7935-4e0f-bd34-4d8a81e14103" width="200">

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Ko--AgentBench-blue)](https://huggingface.co/datasets/huggingface-KREW/Ko-AgentBench)
[![Leaderboard](https://img.shields.io/badge/🏆%20Leaderboard-Ko--AgentBench-green)](https://huggingface.co/spaces/huggingface-KREW/Ko-AgentBench)

</div>

---

## Ko-AgentBench ✨

**한국어 도구 호출(Tool-Calling) 에이전트를 위한 종합 평가 벤치마크**

Ko-AgentBench는 AI 에이전트가 네이버 검색, 카카오맵, 암호화폐 거래소, 주식 조회 등 **한국 사용자가 실제로 사용하는 도구**들을 얼마나 효과적으로 활용하는지 평가합니다.

### 🎯 핵심 특징

- **🇰🇷 한국 특화**: 네이버, 카카오, 티맵, 업비트/빗썸, LS증권, 알라딘 등 한국 서비스 API 활용
- **🔑 API 키 불필요**: 캐시 기반 슈도 API로 실제 API 키 없이도 즉시 평가 가능
- **🎯 7가지 독립 평가**: 도구 선택, 순차/병렬 추론, 오류 처리, 효율성, 장기 맥락 등 다각도 측정
- **🔄 재현 가능**: 동일 조건에서 반복 실행 보장, 연구 재현성 확보

> [!TIP]
> **Why Ko-AgentBench?**
>
> **기존 벤치마크의 한계**
> - 대부분의 도구 호출 벤치마크는 **영어 중심**이며, 한국어 환경과 한국 사용자의 실제 사용 사례를 반영하지 못함
> - 단순히 "정확한 API를 호출했는가"만 평가하고, 실제 업무 흐름의 복잡성(오류 처리, 효율성, 맥락 유지 등)을 고려하지 않음
>
> **Ko-AgentBench의 차별점**
> - **한국 실사용 도구 기반 평가**: 네이버 검색/블로그, 카카오 지도/장소검색, 티맵 경로안내, 업비트/빗썸 암호화폐 거래, LS증권 주식조회, 한국관광공사 축제정보, 알라딘 도서검색 등 한국인이 일상적으로 사용하는 서비스로 구성
> - **현실적인 다중 턴 시나리오**: 단일 API 호출이 아닌, 여러 도구를 연결하고 데이터를 전달하는 실제 업무 흐름 반영
> - **종합적 평가 체계**: 현실성, 명확성, 판별력, 견고성, 효율성, 재현성, 확장성의 7가지 원칙 기반 평가

### 💡 캐시 시스템: API 키 없이 바로 시작

Ko-AgentBench는 **사전 수집된 API 응답 캐시**를 제공하여 실제 API 호출 없이도 벤치마크를 실행할 수 있습니다.

- **Read 모드** (기본): 캐시만 사용, API 키 불필요 → 누구나 즉시 평가 가능
- **Write 모드**: 실제 API 호출 후 캐시 저장 → 새로운 데이터셋 확장 시 사용

```bash
# API 키 없이 실행 (캐시 모드)
uv run run_benchmark_with_logging.py --levels L1 --model openai/gpt-4

# 실제 API 호출 (API 키 필요)
uv run run_benchmark_with_logging.py --cache-mode write
```

---

## 🛠️ 제공되는 API 도구

Ko-AgentBench는 한국 사용자가 실제로 사용하는 다양한 서비스의 API를 제공합니다.

| 서비스 | 도구 | 설명 |
| :--- | :--- | :--- |
| **네이버 검색** | `Search_naver_web`<br>`Search_naver_blog`<br>`Search_naver_news` | 네이버 통합검색, 블로그, 뉴스 검색 API |
| **카카오 로컬** | `AddressToCoord_kakao`<br>`CoordToAddress_kakao`<br>`PlaceSearch_kakao`<br>`CategorySearch_kakao` | 주소-좌표 변환, 장소 검색, 카테고리별 검색 |
| **업비트** | `CryptoPrice_upbit`<br>`MarketList_upbit`<br>`CryptoCandle_upbit` | 암호화폐 현재가, 마켓 목록, 캔들 차트 데이터 조회 |
| **빗썸** | `CryptoPrice_bithumb`<br>`OrderBook_bithumb`<br>`MarketList_bithumb`<br>`CryptoCandle_bithumb` | 암호화폐 현재가, 호가, 마켓 목록, 캔들 차트 조회 |
| **LS증권** | `StockPrice_ls`<br>`MarketIndex_ls`<br>`OrderBook_ls`<br>`SectorStock_ls`<br>`StockTrades_ls` | 국내외 주식 현재가, 시장지수, 호가, 업종별 종목, 체결 내역 조회 |
| **한국투자증권** | `StockPrice_kis`<br>`USStockPrice_kis`<br>`StockChart_kis` | 국내 주식 현재가, 미국 주식 현재가, 차트 데이터 조회 |
| **알라딘** | `ItemSearch_aladin`<br>`ItemList_aladin`<br>`ItemLookup_aladin` | 도서 검색, 베스트셀러/신간 목록, 도서 상세정보 조회 |
| **티맵** | `POISearch_tmap`<br>`Geocoding_tmap`<br>`ReverseGeocoding_tmap`<br>`CarRoute_tmap`<br>`CategorySearch_tmap` | POI 검색, 주소-좌표 변환, 자동차 경로 안내, 카테고리 검색 |
| **네이버 지도** | `Directions_naver` | 대중교통/자동차/도보 경로 안내 |

> **Note**: 모든 API는 캐시 기반으로 동작하며, Read 모드(기본값)에서는 실제 API 키 없이 사용 가능합니다.

---

## �📊 7가지 독립적 평가 차원

에이전트의 도구 호출 능력은 단일 차원으로 측정할 수 없습니다. "도구를 잘 쓴다"는 것은 정확한 도구를 선택하는 능력, 여러 도구를 연결하는 계획 능력, 오류에 대응하는 강건성, 효율적으로 작동하는 능력 등 여러 독립적인 역량의 조합입니다. Ko-AgentBench는 이러한 역량을 7가지로 분리하여 각각 독립적으로 측정합니다. 이는 난이도 단계가 아니라, 서로 다른 측면의 능력을 평가하는 체계입니다.

| 레벨 | Task | 설명 |
| :--- | :--- | :--- |
| **L1** | 단일 도구 호출 | 주어진 단일 도구를 정확한 파라미터로 실행하는 능력 검증 |
| **L2** |  도구 선택 | 여러 도구 중 사용자 요청에 가장 적합한 도구를 선택하는 능력 평가 |
| **L3** |  순차적 추론 | 한 도구의 출력을 다음 도구의 입력으로 사용하는 순차적 계획 및 실행 능력 평가 |
| **L4** |  병렬적 추론 | 여러 도구를 동시에 호출한 후, 그 결과들을 종합하여 결론을 도출하는 능력 평가 |
| **L5** |  오류 처리와 강건성 | API 호출 실패, 정보 부족 등 예외적인 오류 상황에 대처하는 능력 평가 |
| **L6** |  효율적인 도구 활용 | 이전 대화의 도구 호출 결과를 재사용하여 불필요한 반복 실행을 피하는 효율성 평가 |
| **L7** |  장기 컨텍스트 기억 | 긴 대화의 전체 맥락을 기억하고 활용하여 적절한 도구를 호출하는 능력 평가 |

7가지 역량은 각각 독립적으로 측정되며, 종합 점수는 이들의 가중 평균으로 산출됩니다. 이를 통해 모델의 강점과 약점을 세밀하게 파악할 수 있으며, 개선이 필요한 영역을 명확히 식별할 수 있습니다.

---

## 🚀 빠른 시작

### 1) 설치
```bash
# 저장소 클론
git clone https://github.com/Hugging-Face-KREW/Ko-AgentBench
cd Ko-AgentBench

# uv 설치 (미설치 시)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```

### 2) LLM API 키 설정

평가할 모델의 API 키만 설정하면 됩니다 (도구 API 키는 캐시 모드에서 불필요).

```bash
# LLM Model API key (필수)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
```

### 3) 실행과 평가
```bash
# 벤치마크 실행 (L1 레벨, 캐시 읽기 모드)
uv run run_benchmark_with_logging.py --levels L1 --model openai/gpt-4

# 평가 (실행 날짜를 YYYYMMDD 형식으로 입력)
uv run evaluate_model_run.py --date 20251022 --model openai/gpt-4 --format all
```

---

## 📁 프로젝트 구조

### 폴더 구조
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

### 파이프라인 구조

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
uv run run_benchmark_with_logging.py

# 특정 레벨만 실행
uv run run_benchmark_with_logging.py --levels L1,L2,L3

# 특정 모델 지정
uv run run_benchmark_with_logging.py --model openai/gpt-4

# 로컬 모델 사용
uv run run_benchmark_with_logging.py --use-local --model Qwen/Qwen2.5-7B-Instruct

# API 호출 후 캐시 저장
uv run run_benchmark_with_logging.py --cache-mode write

# 태스크 3회 반복 수행
uv run run_benchmark_with_logging.py --repetition 3
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
- `--repetitions`: 반복 실행 횟수 (Pass@k 계산용)
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
uv run evaluate_model_run.py --date 20251022 --model azure/gpt-4o

# 빠른 테스트 (레벨당 1개)
uv run evaluate_model_run.py --date 20251022 --model azure/gpt-4o --quick
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

## 🧩 평가 레벨과 태스크

에이전트의 도구 호출 능력을 7개 레벨로 평가합니다.

| Level | 평가 영역 | 예시 | 주요 지표 |
|-------|----------|------|-----------|
| **L1** | 단일 도구 호출 | "판교역에서 잠실야구장까지 자차로 몇 분 걸릴까?" | ToolAcc, ArgAcc, CallEM, RespOK |
| **L2** | 도구 선택 | "POSCO홀딩스 주식의 현재 호가창을 확인하고 싶어" | SelectAcc |
| **L3** | 도구 순차 추론 | "청량리역 근처 대학교 찾아보고, 그 학교 근처에 병원 몇 개 있는지 조사해줘" | FSM, PSM, ΔSteps_norm, ProvAcc |
| **L4** | 도구 병렬 추론 | "여러 거래소에서 비트코인 시세 동시 조회 후 비교" | Coverage, SourceEPR |
| **L5** | 오류 처리와 강건성 | "아이폰 17 출시일 검색해줘" (API 실패 시 대체 경로) | AdaptiveRoutingScore, FallbackSR |
| **L6** | 효율적인 도구 활용 | "파이썬 알고리즘 트레이딩 책 찾아줘" (중복 호출 방지) | ReuseRate, RedundantCallRate, EffScore |
| **L7** | 장기 컨텍스트 기억 | "요즘 비트코인에 관심이 생겼는데..." (멀티턴 대화) | ContextRetention, RefRecall |

---

## 🧩 평가 지표

### 공통 지표 (모든 레벨)
| 지표 | 설명 | 계산 방식 |
|---|---|---|
| **SR** (Success Rate) | 태스크 완수 점수 | LLM Judge 평가 1-5점 → (점수-1)/4 |
| **EPR/CVR** | 유효 도구 호출 비율 | 유효 호출 수 / 전체 호출 수 |
| **Pass@k** | k회 시도 성공률 | 성공 시도 수 / k |

### 레벨별 전용 지표

**L1: 단일 도구 호출**
| 지표 | 설명 | 계산 방식 |
|---|---|---|
| **ToolAcc** | 올바른 도구 선택 | 일치=1, 불일치=0 |
| **ArgAcc** | 인자 정확도 | LLM Judge 평가 1-5 → 0-1 |
| **CallEM** (Call Exact Match) | 도구+인자 완전 일치 | 0 또는 1 |
| **RespOK** | 응답 형식 준수 | 0 또는 1 |

**L2: 도구 선택**
| 지표 | 설명 | 계산 방식 |
|---|---|---|
| **SelectAcc** | 올바른 도구 선택률 | 0 또는 1 |

**L3: 순차적 추론**
| 지표 | 설명 | 계산 방식 |
|---|---|---|
| **FSM** (Full Sequence Match) | 호출 순서 완전 일치 | 0 또는 1 |
| **PSM** (Partial Sequence Match) | 필수 도구 포함률 | 포함된 필수 도구 / 전체 필수 도구 |
| **ΔSteps_norm** | 효율성 (최소 경로 대비) | min(1, 최소 단계 / 실제 단계) |
| **ProvAcc** | 인자 전달 정확도 | 올바른 데이터 흐름 / 전체 흐름 |

**L4: 병렬적 추론**
| 지표 | 설명 | 계산 방식 |
|---|---|---|
| **Coverage** | 필수 도구 실행률 | 성공한 필수 도구 / 전체 필수 도구 |
| **SourceEPR** | 도구별 유효 호출률 평균 | 평균(유효 호출 / 전체 호출) |

**L5: 오류 처리와 강건성**
| 지표 | 설명 | 계산 방식 |
|---|---|---|
| **AdaptiveRoutingScore** | 실패 후 대체 경로 전환 민첩성 | 1 / (1 + 전환 지연 단계) |
| **FallbackSR** | 대체 경로 성공률 | 대체 성공 / 대체 시도 |

**L6: 효율적인 도구 활용**
| 지표 | 설명 | 계산 방식 |
|---|---|---|
| **ReuseRate** | 재사용률 | 재사용 / (재사용+중복) |
| **RedundantCallRate** | 중복 호출 방지율 | 1 - (중복 호출 / 재사용 기회) |
| **EffScore** | 성공 시 효율 점수 | min(1, 최소 단계 / 실제 단계) |

**L7: 장기 컨텍스트 기억**
| 지표 | 설명 | 계산 방식 |
|---|---|---|
| **ContextRetention** | 맥락 유지 능력 | LLM Judge 평가 1-5 → 0-1 |
| **RefRecall** | 정보 회상 정확도 | LLM Judge 평가 1-5 → 0-1 |

### Judge 평가
- 평가 모델: GPT-4o, Claude, Gemini 앙상블
- 점수 집계: 평균 또는 중앙값
- 모델명을 제거한 블라인드 평가로 공정성 확보

---

## 🧮 종합 점수

- **기본 능력** = L1-L3 지표 평균 (40%)
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

##  사용 예시

```bash
# 1) GPT-4로 L1-L3 레벨 평가
uv run run_benchmark_with_logging.py --levels L1,L2,L3 --model openai/gpt-4

# 2) Claude로 전체 레벨 평가 + 캐시 생성
uv run run_benchmark_with_logging.py --model anthropic/claude-3-5-sonnet-20241022 --cache-mode write

# 3) 로컬 모델 4bit 양자화 평가
uv run run_benchmark_with_logging.py --use-local --model Qwen/Qwen2.5-7B-Instruct --quantization 4bit --device cuda

# 4) 멀티턴 대화 레벨 평가
uv run run_benchmark_with_logging.py --levels L6,L7 --max-steps 20

# 5) 평가 보고서 생성
uv run evaluate_model_run.py --date 20251022 --model azure/gpt-4o --format all

# 6) 빠른 샘플 평가
uv run evaluate_model_run.py --date 20251022 --model azure/gpt-4o --quick
```

---

## ⚖️ 라이선스

[Apache-2.0](LICENSE) 라이선스를 따릅니다.
