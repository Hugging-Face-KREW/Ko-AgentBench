---
name: Bug report
about: 버그 신고 및 최소 재현 정보 제공
title: "[Bug] <summary>"
labels: bug
assignees: harheem, 4N3MONE

---

## ✔️ 사전 확인(필수)
- [ ] 유사/중복 이슈를 검색했습니다.
- [ ] 최신 `main`(또는 관련 브랜치)에서도 재현됨을 확인했습니다.

## 무엇이 발생했나요?
명확한 증상/에러 메시지를 적어주세요. 스크린샷, 로그 일부를 포함할 수 있어요.

## 기대했던 동작
버그가 없었다면 무엇이 일어났어야 하나요?

## 재현 방법(최소/정확하게)
가능한 **최소 재현**을 목표로 해주세요.
```bash
# 예시
conda create -n kab python=3.11
pip install -e .
python tools/run_eval.py \
  --levels 3 \
  --tasks reservation.call \
  --model_provider openai \
  --model gpt-4o-mini \
  --seed 42 \
  --use-cache true
```
재현 결과/오류:

## 영향받은 레벨(Level)
해당되는 항목에 체크해주세요.
- [ ] L1  
- [ ] L2
- [ ] L3
- [ ] L4
- [ ] L5
- [ ] L6  
- [ ] L7  

## 영향받은 태스크/시나리오
예: `navigation.kakao_maps.place_search`, `reservation.call` 등

## 버전/커밋
- Ko-AgentBench 브랜치/커밋 또는 태그: `main @ <SHA>` / `vX.Y.Z`

## 실행 환경
- OS: (예: Ubuntu 22.04 / macOS 14 / Windows 11)
- Python: 3.10 / 3.11
- CUDA/Driver: 12.1 / 550.xx (해당 시)
- GPU: (예: A100 40GB / 로컬/원격)
- 모델/백엔드: (예: openai:gpt-4o-mini / vLLM 0.5.x / transformers 4.xx)
- 중요 라이브러리: torch==2.4, datasets, accelerate …
- 네트워크/프록시: (있는 경우)

## 로그/스택트레이스
민감정보는 마스킹해서 첨부해주세요.

## 추가 정보
스크린샷, 설정 파일, 결과 아티팩트 링크 등
