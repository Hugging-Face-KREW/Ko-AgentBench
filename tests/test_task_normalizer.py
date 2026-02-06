from bench.tasks.task_normalizer import convert_tasks_for_runner, normalize_task_for_runner


def test_normalize_v2_task_derives_v2_fields():
    task = {
        "schema_version": "2.0",
        "task_id": "L3-TEST-V2",
        "track": "A",
        "level": 3,
        "instruction": {"primary": "요약해줘"},
        "available_tools": [
            {"name": "WebSearch_naver", "input_schema": {}},
            {"name": "BlogSearch_naver", "input_schema": {}},
        ],
        "golden": {
            "required_tools": ["WebSearch_naver"],
            "action_graph": {
                "nodes": [
                    {"id": "n1", "tool": "WebSearch_naver", "args": {"query": "겨울 헤어"}},
                    {"id": "n2", "tool": "BlogSearch_naver", "args": {"query": "레이어드"}},
                ],
                "edges": [
                    {
                        "from": "n1",
                        "to": "n2",
                        "mapping": [{"from_field": "items[0].title", "to_arg": "query"}],
                    }
                ],
                "min_steps": 2,
            },
            "expected_answer": {"required_fields": ["items[0].title"]},
        },
        "evaluation": {
            "deterministic_checks": ["TRF1", "ArgF1"],
            "judge_checks": ["SR"],
            "metric_weights": {"TRF1": 0.5, "ArgF1": 0.3, "SR": 0.2},
        },
        "runtime_constraints": {"max_steps": 5, "timeout_sec": 30},
        "metadata": {
            "domain": "search",
            "difficulty": "medium",
            "author": "tester",
            "created_at": "2026-02-06T00:00:00+09:00",
        },
    }

    normalized = normalize_task_for_runner(task)
    assert normalized is not None
    assert normalized["id"] == "L3-TEST-V2"
    assert normalized["description"] == "요약해줘"
    assert normalized["minimum_steps"] == 2
    assert len(normalized["golden_action"]) == 2
    assert normalized["golden_action"][0]["tool"] == "WebSearch_naver"
    assert normalized["data_flow"] == [
        {
            "from_step": 1,
            "from_output": "items[0].title",
            "to_step": 2,
            "to_parameter": "query",
        }
    ]
    assert normalized["expected_output"] == {"required_fields": ["items[0].title"]}
    assert normalized["context_tests"] == []


def test_normalize_v1_task_preserves_v1_fields():
    task = {
        "task_id": "L1-TEST-V1",
        "instruction": "강남역까지 길찾기",
        "task_level": 1,
        "task_category": "single_tool",
        "available_tools": ["Directions_naver"],
        "golden_action": {"tool": "Directions_naver", "args": {"start": "시청", "goal": "강남역"}},
        "resp_schema": {"type": "string"},
        "arg_schema": {"start": {"type": "string"}, "goal": {"type": "string"}},
        "conversation_tracking": {
            "turns": [
                {
                    "turn_number": 1,
                    "role": "assistant",
                    "content": "알겠습니다.",
                    "actions": [{"tool": "CoordToAddress_kakao"}],
                }
            ],
            "evaluation_context": {
                "context_tests": [
                    {
                        "turn": 2,
                        "expected_action": {"tool": "Directions_naver", "args": {"goal": "강남역"}},
                    }
                ]
            },
        },
    }

    normalized = normalize_task_for_runner(task)
    assert normalized is not None
    assert normalized["description"] == "강남역까지 길찾기"
    assert normalized["level"] == 1
    assert normalized["category"] == "single_tool"
    assert normalized["expected_output"] == {"type": "string"}
    assert normalized["golden_action"] == [
        {"tool": "Directions_naver", "args": {"start": "시청", "goal": "강남역"}}
    ]
    assert "Directions_naver" in normalized["available_tools"]
    assert "CoordToAddress_kakao" in normalized["available_tools"]
    assert len(normalized["context_tests"]) == 1


def test_convert_tasks_for_runner_supports_nested_task_lists():
    nested = [[{"task_id": "ONE", "instruction": "x", "available_tools": ["ToolA"]}]]
    out = convert_tasks_for_runner(nested)
    assert len(out) == 1
    assert out[0]["id"] == "ONE"
    assert out[0]["available_tools"] == ["ToolA"]


def test_convert_tasks_for_runner_schema_mode_filters_versions():
    mixed = [
        {"task_id": "V1", "instruction": "a", "available_tools": ["ToolA"]},
        {"task_id": "V2", "schema_version": "2.0", "instruction": {"primary": "b"}, "available_tools": ["ToolB"]},
    ]
    only_v2 = convert_tasks_for_runner(mixed, schema_mode="v2")
    only_v1 = convert_tasks_for_runner(mixed, schema_mode="v1")
    assert [t["id"] for t in only_v2] == ["V2"]
    assert [t["id"] for t in only_v1] == ["V1"]
