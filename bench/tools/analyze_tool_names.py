#!/usr/bin/env python3
"""데이터셋과 tool_catalog 간 도구 이름 불일치 분석"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# tool_catalog에서 직접 데이터 로드 (import 대신 파일 파싱)
def load_catalog_data():
    """tool_catalog.py 파일을 파싱하여 도구 정보 추출"""
    catalog_path = Path(__file__).parent / "tool_catalog.py"
    
    with open(catalog_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # TOOL_CATALOG 키 추출 - 정규식 사용
    catalog_tools = set()
    
    # TOOL_CATALOG 딕셔너리 찾기
    catalog_pattern = r'TOOL_CATALOG:.*?\{(.*?)\n\}'
    catalog_match = re.search(catalog_pattern, content, re.DOTALL)
    
    if catalog_match:
        catalog_content = catalog_match.group(1)
        # 각 도구 이름 추출: "tool_name": ( 형태
        tool_pattern = r'"([^"]+)":\s*\('
        for match in re.finditer(tool_pattern, catalog_content):
            tool_name = match.group(1)
            catalog_tools.add(tool_name)
    
    # TOOL_ALIAS_MAP 추출
    alias_map = {}
    
    alias_pattern = r'TOOL_ALIAS_MAP:.*?\{(.*?)\n\}'
    alias_match = re.search(alias_pattern, content, re.DOTALL)
    
    if alias_match:
        alias_content = alias_match.group(1)
        # 각 별칭 추출: "old_name": "new_name" 형태
        alias_entry_pattern = r'"([^"]+)":\s*"([^"]+)"'
        for match in re.finditer(alias_entry_pattern, alias_content):
            old_name = match.group(1)
            new_name = match.group(2)
            alias_map[old_name] = new_name
    
    return catalog_tools, alias_map

TOOL_CATALOG_KEYS, TOOL_ALIAS_MAP = load_catalog_data()


def extract_tools_from_dataset(json_path: Path) -> Set[str]:
    """데이터셋 파일에서 사용된 모든 도구 이름 추출"""
    tools = set()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        # golden_action에서 도구 추출
        if 'golden_action' in item:
            for action in item['golden_action']:
                if 'tool' in action:
                    tools.add(action['tool'])
        
        # available_tools에서 도구 추출
        if 'available_tools' in item:
            tools.update(item['available_tools'])
        
        # essential_tools에서 도구 추출
        if 'essential_tools' in item:
            tools.update(item['essential_tools'])
    
    return tools


def analyze_all_datasets(data_dir: Path) -> Dict[str, Set[str]]:
    """모든 데이터셋 파일 분석"""
    all_tools_by_level = {}
    
    for json_file in sorted(data_dir.glob("L*.json")):
        level = json_file.stem
        tools = extract_tools_from_dataset(json_file)
        all_tools_by_level[level] = tools
        print(f"✅ {level}.json 분석 완료: {len(tools)}개 도구 발견")
    
    return all_tools_by_level


def categorize_tools(all_tools: Set[str]) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """도구들을 카테고리별로 분류
    
    Returns:
        (정확히_일치, 별칭_존재, 카탈로그_없음, 사용안됨_카탈로그)
    """
    exact_match = set()      # 카탈로그에 정확히 일치하는 이름
    has_alias = set()        # 별칭으로 매핑되는 도구
    not_in_catalog = set()   # 카탈로그에 없는 도구
    
    for tool in all_tools:
        if tool in TOOL_CATALOG_KEYS:
            exact_match.add(tool)
        elif tool in TOOL_ALIAS_MAP:
            has_alias.add(tool)
        else:
            not_in_catalog.add(tool)
    
    # 카탈로그에는 있지만 데이터셋에서 사용되지 않는 도구
    unused_in_catalog = TOOL_CATALOG_KEYS - all_tools
    
    return exact_match, has_alias, not_in_catalog, unused_in_catalog


def print_analysis_report(
    all_tools_by_level: Dict[str, Set[str]],
    exact_match: Set[str],
    has_alias: Set[str],
    not_in_catalog: Set[str],
    unused_in_catalog: Set[str]
):
    """분석 결과 리포트 출력"""
    
    # 모든 도구 합치기
    all_tools = set()
    for tools in all_tools_by_level.values():
        all_tools.update(tools)
    
    print("\n" + "="*80)
    print("도구 이름 불일치 분석 리포트")
    print("="*80)
    
    print(f"\n📊 전체 통계:")
    print(f"  - 데이터셋에서 사용된 도구: {len(all_tools)}개")
    print(f"  - TOOL_CATALOG에 등록된 도구: {len(TOOL_CATALOG_KEYS)}개")
    print(f"  - TOOL_ALIAS_MAP에 등록된 별칭: {len(TOOL_ALIAS_MAP)}개")
    
    print(f"\n✅ 정확히 일치하는 도구 ({len(exact_match)}개):")
    for tool in sorted(exact_match):
        print(f"  - {tool}")
    
    print(f"\n⚠️  별칭으로 매핑되는 도구 ({len(has_alias)}개):")
    for tool in sorted(has_alias):
        mapped_to = TOOL_ALIAS_MAP[tool]
        print(f"  - {tool:30s} → {mapped_to}")
    
    print(f"\n❌ 카탈로그에 없는 도구 ({len(not_in_catalog)}개):")
    if not_in_catalog:
        for tool in sorted(not_in_catalog):
            print(f"  - {tool}")
    else:
        print("  (없음)")
    
    print(f"\n💤 카탈로그에만 있고 데이터셋에서 사용되지 않는 도구 ({len(unused_in_catalog)}개):")
    for tool in sorted(unused_in_catalog):
        print(f"  - {tool}")
    
    # 레벨별 도구 사용 현황
    print(f"\n📋 레벨별 도구 사용 현황:")
    for level in sorted(all_tools_by_level.keys()):
        tools = all_tools_by_level[level]
        print(f"\n  {level} ({len(tools)}개 도구):")
        for tool in sorted(tools):
            status = ""
            if tool in exact_match:
                status = "✅"
            elif tool in has_alias:
                status = f"⚠️  → {TOOL_ALIAS_MAP[tool]}"
            else:
                status = "❌ NOT IN CATALOG"
            print(f"    - {tool:30s} {status}")
    
    # 권장 사항
    print("\n" + "="*80)
    print("🔧 권장 조치 사항:")
    print("="*80)
    
    if has_alias:
        print(f"\n1. 데이터셋 도구 이름 표준화 ({len(has_alias)}개 수정 필요)")
        print("   다음 도구들의 이름을 데이터셋에서 변경하세요:")
        for old_name in sorted(has_alias):
            new_name = TOOL_ALIAS_MAP[old_name]
            print(f"   - {old_name:30s} → {new_name}")
    
    if not_in_catalog:
        print(f"\n2. 누락된 도구 카탈로그에 추가 ({len(not_in_catalog)}개)")
        print("   다음 도구들을 TOOL_CATALOG에 추가하세요:")
        for tool in sorted(not_in_catalog):
            print(f"   - {tool}")
    
    if unused_in_catalog:
        print(f"\n3. 사용되지 않는 카탈로그 항목 검토 ({len(unused_in_catalog)}개)")
        print("   다음 도구들이 카탈로그에만 있고 데이터셋에서 사용되지 않습니다:")
        for tool in sorted(unused_in_catalog):
            print(f"   - {tool}")
    
    # 별칭 제거 가능 여부
    if has_alias:
        print("\n4. 데이터셋 수정 후 다음을 삭제할 수 있습니다:")
        print("   - tool_catalog.py의 TOOL_ALIAS_MAP")
        print("   - tool_catalog.py의 normalize_tool_name() 함수")
        print("   - resolve_tool_classes()에서 정규화 로직")


def main():
    """메인 함수"""
    # 경로 설정
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent.parent
    data_dir = project_dir / "data"
    
    print("🔍 Ko-AgentBench 도구 이름 분석 시작...")
    print(f"📂 데이터 디렉토리: {data_dir}")
    
    # 데이터셋 분석
    all_tools_by_level = analyze_all_datasets(data_dir)
    
    # 모든 도구 합치기
    all_tools = set()
    for tools in all_tools_by_level.values():
        all_tools.update(tools)
    
    # 도구 분류
    exact_match, has_alias, not_in_catalog, unused_in_catalog = categorize_tools(all_tools)
    
    # 리포트 출력
    print_analysis_report(
        all_tools_by_level,
        exact_match,
        has_alias,
        not_in_catalog,
        unused_in_catalog
    )
    
    print("\n" + "="*80)
    print("✅ 분석 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
