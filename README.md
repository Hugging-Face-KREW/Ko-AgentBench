# Ko-AgentBench

Ko-AgentBench는 도구 호출(함수 호출)을 활용하는 에이전트를 한국어 데이터셋 L1~L7로 평가하는 벤치마크입니다. 본 문서는 벤치마크 실행 스크립트와 플래그 사용법을 설명합니다.

## 요구 사항
- Python 3.10 이상
- [uv](https://github.com/astral-sh/uv) (권장)
- LLM API 키(중 하나 이상): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `AZURE_API_KEY`(+ `AZURE_API_BASE`, `AZURE_API_VERSION`), `HUGGINGFACE_API_KEY`

## 환경 세팅 (uv 권장)
```bash
# Python 설치 (최초 1회)
uv python install 3.10

# 의존성 설치 및 가상환경 생성(.venv)
uv sync --python 3.10

# (선택) .env 사용 시 .env 파일에 API 키 설정
# 예) OPENAI_API_KEY=sk-...
```

pip를 선호한다면 다음과 같이도 가능합니다.
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터셋 경로
기본적으로 `data/` 폴더의 `L1.json` ~ `L7.json`을 자동으로 탐색합니다.
- L6, L7은 멀티턴(`conversation_tracking`) 데이터셋이며, 러너가 자동으로 대화 이력을 주입하고 마지막 사용자 턴까지만 잘라서 모델을 호출합니다.

## 기본 실행
벤치마크 실행 스크립트는 `run_benchmark_with_logging.py` 입니다.
```bash
uv run python run_benchmark_with_logging.py
```

첫 실행 시 스크립트는 환경 변수에 설정된 API 키를 확인하고, 사용 가능한 모델을 자동 선택합니다. 자동 탐지가 어렵다면 아래의 `--model` 옵션을 사용해 명시적으로 지정할 수 있습니다.

## CLI 플래그
`run_benchmark_with_logging.py`는 다음 옵션을 제공합니다.

- `--levels L6,L7`
	- 실행할 레벨을 콤마로 구분해 지정합니다. 미설정 시 `data/`에서 감지된 모든 레벨을 실행합니다.
// `--limit` 옵션은 제거되었습니다. 항상 레벨의 모든 태스크를 실행합니다.
- `--max-steps 10`
	- 태스크별 최대 스텝 수를 지정합니다. 도구 호출 루프의 상한입니다.
- `--timeout 60`
	- 태스크별 타임아웃(초)을 지정합니다.
- `--no-save-logs`
	- 실행 결과 JSON 로그 저장을 비활성화합니다(기본값: 저장).
- `--model provider/model`
	- 사용할 모델을 명시합니다. 예: `openai/gpt-4.1`, `anthropic/claude-3-5-sonnet-20241022`, `google/gemini-1.5-pro` 등.

## 사용 예시
- 멀티턴 레벨(L6,L7)만 실행
```bash
uv run python run_benchmark_with_logging.py --levels L6,L7 --max-steps 4 --timeout 60
```

- 로그 저장 없이 가볍게 테스트
```bash
uv run python run_benchmark_with_logging.py --levels L7 --no-save-logs
```

- 모델을 명시적으로 지정(환경 변수와 무관하게 강제)
```bash
uv run python run_benchmark_with_logging.py --levels L7 --model openai/gpt-4.1
```

## 결과 로그
- 기본 저장 위치: `logs/benchmark_results/`
- 파일명: `<레벨>_<모델>_<타임스탬프>.json`
- 포함 내용:
	- 메타데이터(성공률, 총 소요 시간, 스텝/도구 호출 통계 등)
	- 태스크별 요약(성공 여부, 도구 호출 내역, 최종 응답)

## 평가 및 멀티턴 처리 동작
- 멀티턴(L6/L7): `conversation_tracking.turns`의 사용자/어시스턴트 메시지가 컨텍스트로 주입됩니다. 다음 어시스턴트 턴에 대응하기 위해 마지막 사용자 턴까지만 포함합니다.
- 평가(Judge): 도구를 사용하는 태스크의 경우, 최종 어시스턴트 텍스트보다 "최근 성공한 도구 결과"를 우선적으로 실제 출력으로 간주하여 매칭합니다.

## 트러블슈팅
- "No LLM API keys found" 경고가 뜨는 경우
	- 환경 변수 또는 `.env`에 API 키를 설정하세요. 예: `export OPENAI_API_KEY=...`
- 특정 제공자 모델을 쓰고 싶지만 자동 선택이 안 되는 경우
	- `--model provider/model`로 명시하세요.
- 실행이 오래 걸려 중단(Ctrl+C, exit code 130)되는 경우
	- `--limit`과 `--max-steps`를 줄여 빠르게 확인하세요.
- "Missing tool in catalog" 경고
	- 데이터셋의 도구 이름과 레지스트리 매핑(별칭→실제 도구명) 확인이 필요할 수 있습니다.

## 로컬 스모크 테스트
외부 API 호출 없이 러너 동작을 빠르게 확인하려면 다음을 참고하세요.
```bash
uv run python -c "from bench.runner.test_run import run_simple_test; run_simple_test()"
```
간단한 모의 도구와 어댑터로 툴 호출 루프와 평가 흐름을 점검합니다.
