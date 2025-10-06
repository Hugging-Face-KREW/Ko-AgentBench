"""
Tool Call 결과 분석 스크립트

L1~L6 벤치마크 결과 로그 파일들을 분석하여
각 tool별로 호출 횟수, 성공률, 실패율 등을 집계합니다.
"""

import json
import glob
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_all_benchmark_results(log_dir: str = "logs/benchmark_results") -> Dict[str, Any]:
    """모든 벤치마크 결과 파일 로드"""
    results = {}
    json_files = sorted(glob.glob(f"{log_dir}/L*.json"))
    
    for filepath in json_files:
        level_name = Path(filepath).stem.split('_')[0]  # L1, L2, etc.
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results[level_name] = data
                print(f"✓ Loaded {filepath}")
        except Exception as e:
            print(f"✗ Failed to load {filepath}: {e}")
    
    return results


def analyze_tool_usage(all_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Tool별 사용 통계 분석"""
    tool_stats = defaultdict(lambda: {
        'total_calls': 0,
        'successful_calls': 0,
        'failed_calls': 0,
        'success_rate': 0.0,
        'levels_used': set(),
        'tasks_used': [],
        'error_messages': []
    })
    
    for level_name, level_data in all_results.items():
        results = level_data.get('results', [])
        
        for task_result in results:
            task_id = task_result.get('task_id', 'unknown')
            tool_invocations = task_result.get('tool_invocations', [])
            
            for invocation in tool_invocations:
                tool_name = invocation.get('tool_name', 'unknown')
                success = invocation.get('success', False)
                error = invocation.get('error')
                
                # 통계 업데이트
                tool_stats[tool_name]['total_calls'] += 1
                tool_stats[tool_name]['levels_used'].add(level_name)
                tool_stats[tool_name]['tasks_used'].append({
                    'task_id': task_id,
                    'level': level_name,
                    'success': success,
                    'arguments': invocation.get('arguments', {}),
                    'error': error
                })
                
                if success:
                    tool_stats[tool_name]['successful_calls'] += 1
                else:
                    tool_stats[tool_name]['failed_calls'] += 1
                    if error:
                        tool_stats[tool_name]['error_messages'].append({
                            'task_id': task_id,
                            'error': error
                        })
    
    # Success rate 계산 및 set을 list로 변환
    for tool_name, stats in tool_stats.items():
        total = stats['total_calls']
        if total > 0:
            stats['success_rate'] = round(stats['successful_calls'] / total * 100, 2)
        stats['levels_used'] = sorted(list(stats['levels_used']))
    
    return dict(tool_stats)


def analyze_by_level(all_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Level별 통계 분석"""
    level_stats = {}
    
    for level_name, level_data in all_results.items():
        metadata = level_data.get('metadata', {})
        tool_usage = level_data.get('tool_usage_statistics', {})
        
        level_stats[level_name] = {
            'total_tasks': metadata.get('total_tasks', 0),
            'successful_tasks': metadata.get('successful_tasks', 0),
            'failed_tasks': metadata.get('failed_tasks', 0),
            'success_rate': metadata.get('success_rate', 0.0),
            'total_tool_calls': metadata.get('total_tool_calls', 0),
            'average_tool_calls': metadata.get('average_tool_calls', 0.0),
            'tools_used': tool_usage
        }
    
    return level_stats


def generate_summary_report(tool_stats: Dict[str, Dict[str, Any]], 
                           level_stats: Dict[str, Dict[str, Any]]) -> str:
    """요약 리포트 생성"""
    lines = []
    lines.append("=" * 100)
    lines.append("TOOL CALL ANALYSIS REPORT")
    lines.append("=" * 100)
    lines.append("")
    
    # Level별 요약
    lines.append("📊 LEVEL SUMMARY")
    lines.append("-" * 100)
    lines.append(f"{'Level':<10} {'Total Tasks':<15} {'Success':<10} {'Failed':<10} {'Success Rate':<15} {'Tool Calls':<15}")
    lines.append("-" * 100)
    
    total_tasks = 0
    total_success = 0
    total_failed = 0
    total_tool_calls = 0
    
    for level_name in sorted(level_stats.keys()):
        stats = level_stats[level_name]
        total_tasks += stats['total_tasks']
        total_success += stats['successful_tasks']
        total_failed += stats['failed_tasks']
        total_tool_calls += stats['total_tool_calls']
        
        lines.append(
            f"{level_name:<10} {stats['total_tasks']:<15} {stats['successful_tasks']:<10} "
            f"{stats['failed_tasks']:<10} {stats['success_rate']:<15.2f} {stats['total_tool_calls']:<15}"
        )
    
    lines.append("-" * 100)
    overall_success_rate = (total_success / total_tasks * 100) if total_tasks > 0 else 0
    lines.append(
        f"{'TOTAL':<10} {total_tasks:<15} {total_success:<10} {total_failed:<10} "
        f"{overall_success_rate:<15.2f} {total_tool_calls:<15}"
    )
    lines.append("")
    lines.append("")
    
    # Tool별 요약
    lines.append("🔧 TOOL USAGE SUMMARY")
    lines.append("-" * 100)
    lines.append(f"{'Tool Name':<30} {'Calls':<10} {'Success':<10} {'Failed':<10} {'Success Rate':<15} {'Levels':<20}")
    lines.append("-" * 100)
    
    # 호출 횟수로 정렬
    sorted_tools = sorted(tool_stats.items(), key=lambda x: x[1]['total_calls'], reverse=True)
    
    for tool_name, stats in sorted_tools:
        levels = ', '.join(stats['levels_used'])
        lines.append(
            f"{tool_name:<30} {stats['total_calls']:<10} {stats['successful_calls']:<10} "
            f"{stats['failed_calls']:<10} {stats['success_rate']:<15.2f} {levels:<20}"
        )
    
    lines.append("")
    lines.append("")
    
    # 실패한 Tool 상세 정보
    lines.append("❌ FAILED TOOL CALLS DETAIL")
    lines.append("-" * 100)
    
    failed_tools = {name: stats for name, stats in tool_stats.items() if stats['failed_calls'] > 0}
    
    if failed_tools:
        for tool_name, stats in sorted(failed_tools.items(), key=lambda x: x[1]['failed_calls'], reverse=True):
            lines.append(f"\n{tool_name} (Failed: {stats['failed_calls']}/{stats['total_calls']})")
            lines.append("-" * 80)
            
            # 에러 메시지 표시 (최대 5개)
            for error_info in stats['error_messages'][:5]:
                lines.append(f"  Task: {error_info['task_id']}")
                error_msg = error_info['error']
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                lines.append(f"  Error: {error_msg}")
                lines.append("")
            
            if len(stats['error_messages']) > 5:
                lines.append(f"  ... and {len(stats['error_messages']) - 5} more errors")
                lines.append("")
    else:
        lines.append("No failed tool calls! 🎉")
    
    lines.append("")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def save_detailed_analysis(tool_stats: Dict[str, Dict[str, Any]], 
                          level_stats: Dict[str, Dict[str, Any]],
                          output_file: str = "logs/tool_call_analysis.json"):
    """상세 분석 결과를 JSON으로 저장"""
    analysis_data = {
        "summary": {
            "total_unique_tools": len(tool_stats),
            "total_tool_calls": sum(stats['total_calls'] for stats in tool_stats.values()),
            "total_successful_calls": sum(stats['successful_calls'] for stats in tool_stats.values()),
            "total_failed_calls": sum(stats['failed_calls'] for stats in tool_stats.values()),
        },
        "tool_statistics": tool_stats
        # level_statistics는 제외
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    return output_file


def main():
    """메인 실행 함수"""
    print("=" * 100)
    print("Ko-AgentBench Tool Call Analysis")
    print("=" * 100)
    print()
    
    # 결과 로드
    print("Loading benchmark results...")
    all_results = load_all_benchmark_results()
    
    if not all_results:
        print("✗ No benchmark results found!")
        return
    
    print(f"\n✓ Loaded {len(all_results)} levels: {', '.join(sorted(all_results.keys()))}")
    print()
    
    # 분석 수행
    print("Analyzing tool usage...")
    tool_stats = analyze_tool_usage(all_results)
    
    print("Analyzing level statistics...")
    level_stats = analyze_by_level(all_results)
    
    # 리포트 생성 및 출력
    print("\n" + "=" * 100)
    report = generate_summary_report(tool_stats, level_stats)
    print(report)
    
    # 상세 결과 저장
    output_file = save_detailed_analysis(tool_stats, level_stats)
    print(f"\n✓ Detailed analysis saved to: {output_file}")
    
    # 텍스트 리포트도 저장
    report_file = "logs/tool_call_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ Text report saved to: {report_file}")


if __name__ == "__main__":
    main()
