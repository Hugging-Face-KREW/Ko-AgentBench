# Ko-AgentBench 평가 보고서

## 실행 정보

- **평가 대상 모델**: azure/gpt-5
- **Judge 모델**: ['azure/gpt-4o', 'anthropic/claude-sonnet-4-5-20250929', 'gemini/gemini-2.5-pro-preview-03-25']
- **실행 날짜**: 20251021
- **평가 날짜**: 2025-10-21
- **총 태스크**: 160/160

## 레벨별 성능

### Level 1: 단일 도구 호출

- 태스크 수: 16/16
- 성공률: 93.8%
- 평균 실행시간: 17.90초

**메트릭 점수:**

- ArgAcc: 0.208
- CallEM: 0.250
- EPR_CVR: 0.875
- RespOK: 0.000
- SR: 0.938
- ToolAcc: 0.875
- pass@k: 0.938

### Level 2: 도구 선택

- 태스크 수: 56/56
- 성공률: 98.2%
- 평균 실행시간: 24.46초

**메트릭 점수:**

- EPR_CVR: 0.946
- SR: 0.982
- SelectAcc: 0.946
- pass@k: 0.982

### Level 3: 멀티스텝 추론

- 태스크 수: 27/27
- 성공률: 100.0%
- 평균 실행시간: 22.51초

**메트릭 점수:**

- EPR_CVR: 0.759
- FSM: 0.407
- PSM: 0.673
- ProvAcc: 0.444
- SR: 1.000
- pass@k: 1.000
- ΔSteps_norm: 0.537

### Level 4: 멀티소스 통합

- 태스크 수: 10/10
- 성공률: 80.0%
- 평균 실행시간: 36.34초

**메트릭 점수:**

- Coverage: 0.150
- EPR_CVR: 0.200
- SR: 0.800
- SourceEPR: 0.125
- pass@k: 0.800

### Level 5: 오류 처리

- 태스크 수: 26/26
- 성공률: 100.0%
- 평균 실행시간: 26.12초

**메트릭 점수:**

- EPR_CVR: 0.000
- ErrorDetect: 0.212
- FallbackSR: 0.000
- GracefulFail: 0.000
- SR: 1.000
- pass@k: 1.000

### Level 6: 컨텍스트 재사용

- 태스크 수: 15/15
- 성공률: 100.0%
- 평균 실행시간: 66.71초

**메트릭 점수:**

- EPR_CVR: 1.000
- EffScore: 0.000
- RedundantCallRate: 1.000
- ReuseRate: 0.067
- SR: 1.000
- pass@k: 1.000

### Level 7: 멀티턴 대화

- 태스크 수: 10/10
- 성공률: 100.0%
- 평균 실행시간: 51.67초

**메트릭 점수:**

- ContextRetention: 0.025
- EPR_CVR: 1.000
- RefRecall: 0.025
- SR: 1.000
- pass@k: 1.000

