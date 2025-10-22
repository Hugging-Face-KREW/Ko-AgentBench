# 평가 가이드

## 사용법

```bash
# 기본 평가
python evaluate_model_run.py --date 20251022 --model azure/gpt-4o

# 빠른 테스트
python evaluate_model_run.py --date 20251022 --model azure/gpt-4o --quick
```

## 주요 옵션

- `--date`: 벤치마크 실행 날짜 (예: 20251022)
- `--model`: 평가 대상 모델 (예: azure/gpt-4o)
- `--judge-models`: Judge 모델들 (기본: gpt-4o, claude-sonnet, gemini)
- `--sample N`: 레벨당 N개만 평가
- `--quick`: 레벨당 1개만 평가
- `--format`: 출력 형식 (json/csv/markdown/all)

## 환경 설정

`.env` 파일에 API 키 설정:

```bash
AZURE_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GEMINI_API_KEY=your_key
```

## 출력

결과는 `reports/{model}_{date}/` 에 저장됩니다.
