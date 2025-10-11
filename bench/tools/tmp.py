"""간단한 tool calling 예제

1) 모델이 tool_calls 를 반환하면 각 tool 을 실행 (현재 실제 API 미구현 → mock)
2) 실행 결과를 messages 에 tool 역할로 append 후 재호출
3) 최종 assistant 메시지 출력

환경 변수:
  LLM_API_KEY : litellm 에 전달할 키 (필수)
"""

import os
import json
import traceback
import litellm
from naver_search import NaverSearchAPI
from daum_search import DaumSearchAPI

# ---- 환경 변수에서 키 읽기 ----

naver_api = NaverSearchAPI()
daum_api = DaumSearchAPI()

messages = [
    {"role": "user", "content": "삼성전자 주가 알려줘"}
]

tools = [
    naver_api.WebSearch_naver(),
    daum_api.WebSearch_daum()
]

def execute_local_tool(tool_name: str, arguments: dict):
    try:
        if tool_name == "WebSearch_naver":
            query = arguments.get("query")
            return {"engine": "naver", "query": query, "results": ["(임시) 네이버 웹 검색 결과1", "(임시) 결과2"]}
        if tool_name == "WebSearch_daum":
            query = arguments.get("query")
            return {"engine": "daum", "query": query, "results": ["(임시) 다음 웹 검색 결과1", "(임시) 결과2"]}
        return {"error": f"알 수 없는 tool: {tool_name}"}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

def first_call():
    return litellm.completion(
        model="gpt-4.1",
        api_key="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",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

def second_call():
    return litellm.completion(
        model="gpt-4.1",
        api_key="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",
        messages=messages,
    )

# 1차 호출
response = first_call()
assistant_msg = response.choices[0].message
tool_calls = getattr(assistant_msg, "tool_calls", None)

if tool_calls:
    # (중요) tool_messages 이전에 assistant(tool_calls) 메시지를 대화 히스토리에 append
    assistant_msg_dict = {
        "role": assistant_msg.role,
        "content": assistant_msg.content,
        "tool_calls": []
    }
    for tc in tool_calls:
        assistant_msg_dict["tool_calls"].append({
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
        })
    print("=== tool_calls ===")
    print(json.dumps(assistant_msg_dict, ensure_ascii=False, indent=2))
    messages.append(assistant_msg_dict)

    # 각 tool 실행 후 tool 메시지 추가
    for tc in tool_calls:
        fn_name = tc.function.name
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {"raw": tc.function.arguments}
        result = execute_local_tool(fn_name, args)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": fn_name,
            "content": json.dumps(result, ensure_ascii=False)
        })

    # 2차 호출 (tool 결과 반영 최종 답변)
    followup = second_call()
    final_message = followup.choices[0].message
else:
    final_message = assistant_msg

print("=== 최종 모델 응답 ===")
print(final_message)