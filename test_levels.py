"""각 레벨별 데이터 예시 추론 및 로깅 테스트 스크립트"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Type
from dotenv import load_dotenv

from bench.tools.tool_registry import ToolRegistry
from bench.adapters.litellm_adapter import LiteLLMAdapter
from bench.runner import BenchmarkRunner, Judge
from bench.tasks.task_loader import TaskLoader
from bench.tools.base_api import BaseTool
from bench.models import MODEL_IDS
from bench.tools.tool_catalog import resolve_tool_classes, TOOL_CATALOG
from bench.observability import log_status


def run_multiturn_task(task: Dict[str, Any], runner: BenchmarkRunner, registry: ToolRegistry) -> Dict[str, Any]:
    """멀티턴 대화가 있는 태스크를 실행합니다.
    
    Args:
        task: conversation_tracking이 있는 태스크
        runner: BenchmarkRunner 인스턴스
        registry: ToolRegistry 인스턴스
        
    Returns:
        실행 결과
    """
    import time
    
    start_time = time.time()
    conv_tracking = task.get('conversation_tracking', {})
    turns = conv_tracking.get('turns', [])
    
    print(f"\n[멀티턴] 대화 모드 (총 {len(turns)}개 턴)")
    
    # 대화 메시지 초기화
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can use tools to complete tasks."}
    ]
    
    # 사용 가능한 툴 스키마 가져오기
    requested_tools = task.get("available_tools", [])
    
    # available_tools가 없으면 golden_action에서 툴 추출
    if not requested_tools and task.get('golden_action'):
        for action in task['golden_action']:
            tool_name = action.get('tool')
            if tool_name and tool_name != 'reuse' and tool_name not in requested_tools:
                requested_tools.append(tool_name)
    
    available_tools = []
    for tool_name in requested_tools:
        tool = registry.get_tool(tool_name)
        if tool:
            available_tools.append(tool.get_schema())
    
    # 각 턴 실행
    all_tool_calls = []
    turn_results = []
    user_turn_count = 0
    
    for turn in turns:
        if turn.get('role') != 'user':
            continue
        
        user_turn_count += 1
        user_content = turn.get('content', '')
        
        print(f"\n  턴 {user_turn_count}: {user_content[:80]}{'...' if len(user_content) > 80 else ''}")
        
        # 사용자 메시지 추가
        messages.append({"role": "user", "content": user_content})
        
        # LLM 호출
        try:
            response = runner._call_llm_with_retry(messages, available_tools)
            message = response.get('message', {})
            
            # 어시스턴트 응답 추가
            messages.append(message)
            
            # 툴 호출 처리
            turn_tool_calls = []
            if 'tool_calls' in message and message['tool_calls']:
                for tool_call in message['tool_calls']:
                    tool_result = runner._execute_tool_call(tool_call)
                    turn_tool_calls.append(tool_result)
                    all_tool_calls.append({
                        "step": user_turn_count,
                        "tool_call_id": tool_result.get('tool_call_id'),
                        "tool_name": tool_result.get('tool_name'),
                        "arguments": tool_result.get('arguments'),
                        "result": tool_result.get('result'),
                        "success": tool_result.get('success'),
                        "error": tool_result.get('error'),
                    })
                    
                    # 툴 결과를 대화에 추가
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "content": json.dumps(tool_result['result']) if tool_result.get('success') else json.dumps({"error": tool_result.get('error')})
                    })
                
                print(f"    [툴] 호출: {len(turn_tool_calls)}개")
            
            turn_results.append({
                "turn": user_turn_count,
                "success": True,
                "tool_calls": len(turn_tool_calls)
            })
            
        except Exception as e:
            print(f"    [오류] 에러: {str(e)}")
            turn_results.append({
                "turn": user_turn_count,
                "success": False,
                "error": str(e)
            })
    
    execution_time = time.time() - start_time
    
    # 결과 구성
    result = {
        "steps": [],
        "final_response": messages[-1].get('content', '') if messages else '',
        "conversation": messages
    }
    
    # Judge 평가
    evaluation = runner.judge.evaluate(task, result)
    
    return {
        "task_id": task.get('task_id', 'unknown'),
        "success": evaluation.get('success', False),
        "result": result,
        "tool_invocations": all_tool_calls,
        "evaluation": evaluation,
        "execution_time": execution_time,
        "steps_taken": user_turn_count,
        "multiturn": True,
        "total_turns": user_turn_count,
        "successful_turns": sum(1 for t in turn_results if t.get('success', False)),
        "error": None
    }


def test_level_samples(
    model_name: str = "huggingface/Qwen/Qwen3-4B-Instruct-2507",
    save_logs: bool = True,
    log_dir: str = "logs/level_tests",
):
    """각 레벨별로 첫 번째 태스크를 실행하여 추론 및 로깅을 테스트합니다.
    
    Args:
        model_name: 사용할 모델 이름
        save_logs: 로그 저장 여부
        log_dir: 로그 저장 디렉토리
    """
    
    print("="*80)
    print("Ko-AgentBench 레벨별 추론 및 로깅 테스트")
    print("="*80)
    print(f"모델: {model_name}")
    print(f"로그 디렉토리: {log_dir}")
    print("="*80 + "\n")
    
    # 로그 디렉토리 생성
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 각 레벨별 데이터 파일 경로
    data_dir = Path(__file__).parent / "data"
    level_files = [
        ("L1", data_dir / "L1.json"),
        ("L2", data_dir / "L2.json"),
        ("L3", data_dir / "L3.json"),
        ("L4", data_dir / "L4.json"),
        ("L5", data_dir / "L5.json"),
        ("L6", data_dir / "L6.json"),
        ("L7", data_dir / "L7.json"),
    ]
    
    all_results = []
    
    for level_name, level_file in level_files:
        if not level_file.exists():
            print(f"[경고] {level_name} 파일을 찾을 수 없습니다: {level_file}")
            continue
            
        print(f"\n{'='*80}")
        print(f"[레벨] {level_name} 테스트 시작")
        print(f"{'='*80}")
        
        try:
            # 태스크 로드
            task_loader = TaskLoader()
            tasks = task_loader.load_tasks(str(level_file))
            
            if not tasks:
                print(f"[경고] {level_name}에 태스크가 없습니다.")
                continue
            
            # 첫 번째 태스크만 테스트
            task = tasks[0]
            print(f"\n[태스크] Task ID: {task.get('task_id', 'unknown')}")
            print(f"   Level: {task.get('task_level', 'unknown')}")
            print(f"   Category: {task.get('task_category', 'unknown')}")
            print(f"   Instruction: {task.get('instruction', 'N/A')}")
            
            # 태스크에 필요한 툴 수집
            requested_tools = task.get("available_tools", [])
            
            # available_tools가 없으면 golden_action에서 툴 추출
            if not requested_tools and task.get('golden_action'):
                for action in task['golden_action']:
                    tool_name = action.get('tool')
                    if tool_name and tool_name != 'reuse' and tool_name not in requested_tools:
                        requested_tools.append(tool_name)
            
            print(f"\n[툴] 필요한 툴: {', '.join(requested_tools) if requested_tools else '없음'}")
            
            # 툴 클래스 해결
            tool_classes = resolve_tool_classes(requested_tools) if requested_tools else []
            missing = [n for n in requested_tools if n not in TOOL_CATALOG]
            if missing:
                print(f"[경고] 등록할 수 없는 툴: {', '.join(missing)}")
            
            # 컴포넌트 설정
            registry = ToolRegistry()
            for tool_class in tool_classes:
                registry.register_tool(tool_class)
            
            adapter = LiteLLMAdapter(model_name)
            judge = Judge(llm_adapter=adapter)
            runner = BenchmarkRunner(adapter, registry, judge, max_steps=5, timeout=60)
            
            # 멀티턴 대화가 있는지 확인
            has_multiturn = 'conversation_tracking' in task and task['conversation_tracking'].get('turns')
            
            # 태스크 실행
            if has_multiturn:
                print(f"\n[실행] 멀티턴 대화 추론 실행 중...")
                result = run_multiturn_task(task, runner, registry)
            else:
                print(f"\n[실행] 추론 실행 중...")
                result = runner.run_task(task)
            
            # 결과 출력
            print(f"\n{'='*80}")
            print(f"[완료] 실행 완료")
            print(f"{'='*80}")
            print(f"성공 여부: {'[성공]' if result['success'] else '[실패]'}")
            print(f"실행 시간: {result['execution_time']:.2f}초")
            
            if result.get('multiturn'):
                print(f"모드: [멀티턴] 대화")
                print(f"턴 수: {result.get('successful_turns', 0)}/{result.get('total_turns', 0)} 성공")
            else:
                print(f"단계 수: {result['steps_taken']}")
            
            # 툴 호출 정보
            tool_invocations = result.get('tool_invocations', [])
            if tool_invocations:
                print(f"\n[툴] 호출 내역 ({len(tool_invocations)}개):")
                for idx, inv in enumerate(tool_invocations, 1):
                    status = "[성공]" if inv.get('success') else "[실패]"
                    print(f"  {idx}. {status} {inv.get('tool_name', 'unknown')}")
                    print(f"     인자: {inv.get('arguments', {})}")
                    if inv.get('error'):
                        print(f"     에러: {inv.get('error')}")
            
            # 최종 응답
            if result.get('result') and result['result'].get('final_response'):
                response = result['result']['final_response']
                print(f"\n[응답] 최종 응답:")
                print(f"   {response[:200]}{'...' if len(response) > 200 else ''}")
            
            # 평가 결과
            evaluation = result.get('evaluation', {})
            if evaluation:
                print(f"\n[평가] 평가 결과:")
                print(f"   판정: {'[통과]' if evaluation.get('success') else '[실패]'}")
                metrics = evaluation.get('metrics', {})
                if metrics:
                    print(f"   메트릭:")
                    for metric_name, metric_data in metrics.items():
                        print(f"     - {metric_name}: {metric_data.get('score', 'N/A')}")
            
            # 에러 정보
            if result.get('error'):
                print(f"\n[오류] 에러: {result['error']}")
            
            all_results.append({
                "level": level_name,
                "result": result
            })
            
        except Exception as e:
            print(f"\n[오류] {level_name} 테스트 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "level": level_name,
                "result": {
                    "task_id": task.get('task_id', 'unknown') if 'task' in locals() else 'unknown',
                    "success": False,
                    "error": str(e),
                    "execution_time": 0,
                    "steps_taken": 0,
                }
            })
    
    # 전체 결과 저장
    if save_logs and all_results:
        summary_file = Path(log_dir) / "level_test_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"[저장] 전체 테스트 결과 저장 완료")
        print(f"파일 위치: {summary_file}")
        print(f"{'='*80}")
    
    # 전체 요약
    print(f"\n{'='*80}")
    print(f"[요약] 전체 테스트 요약")
    print(f"{'='*80}")
    total = len(all_results)
    successful = sum(1 for r in all_results if r['result'].get('success', False))
    print(f"전체 레벨: {total}")
    print(f"성공: {successful}")
    print(f"실패: {total - successful}")
    print(f"성공률: {(successful/total*100):.1f}%" if total > 0 else "N/A")
    print(f"{'='*80}\n")


def main():
    """메인 함수"""
    load_dotenv()
    log_status()
    
    # API 키 확인
    provider_keys = [
        "HUGGINGFACE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]
    
    has_key = False
    for key in provider_keys:
        if os.getenv(key):
            print(f"[OK] {key} 설정됨")
            has_key = True
        else:
            print(f"[경고] {key} 미설정")
    
    if not has_key:
        print("\n[오류] 에러: LLM API 키가 설정되지 않았습니다.")
        print("   .env 파일에 API 키를 설정해주세요.")
        return
    
    print()
    
    # 사용 가능한 모델 확인
    if MODEL_IDS:
        print(f"사용 가능한 모델: {', '.join(MODEL_IDS)}")
        selected_model = MODEL_IDS[0]
        print(f"선택된 모델: {selected_model}\n")
        
        # 테스트 실행
        test_level_samples(
            model_name=selected_model,
            save_logs=True,
            log_dir="logs/level_tests"
        )
    else:
        print("[오류] 에러: 사용 가능한 모델이 없습니다.")


if __name__ == "__main__":
    main()

