from cypher_3dsg_agent_distillation.agent_records import DEFAULT_AGENT_TOOL


def answerable_trace(**overrides):
    payload = {
        "uid": "one",
        "scene_id": "scene",
        "question": "How many chairs?",
        "task_type": "object_class_count",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": DEFAULT_AGENT_TOOL,
                            "arguments": {
                                "cypher_string": "MATCH (o:Object) RETURN count(o) AS count"
                            },
                        }
                    }
                ],
            },
            {"role": "tool", "tool_name": DEFAULT_AGENT_TOOL, "content": "[{'count': 3}]"},
            {"role": "assistant", "content": "<answer>3</answer>"},
        ],
        "gold_cypher": "MATCH (o:Object) RETURN count(o) AS count",
        "expected_solution": 3,
        "validation": {"static_valid": True, "executable": True, "solution_match": True},
    }
    payload.update(overrides)
    return payload
