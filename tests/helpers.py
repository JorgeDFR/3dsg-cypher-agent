from cypher_3dsg_agent_distillation.agents.agent_records import DEFAULT_AGENT_TOOL


def answerable_trace(**overrides):
    payload = {
        "uid": "one",
        "scene_id": "scene",
        "question": "How many chairs?",
        "task_type": "object_class_count",
        "task_family": "cypher_qa",
        "output_type": "SLDP",
        "validation_contract": qa_contract(["int"]),
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
            {
                "role": "tool",
                "tool_name": DEFAULT_AGENT_TOOL,
                "content": '{"ok": true, "rows": [{"count": 3}]}',
            },
            {"role": "assistant", "content": "<answer>3</answer>"},
        ],
        "gold_cypher": "MATCH (o:Object) RETURN count(o) AS count",
        "expected_solution": 3,
        "validation": {"static_valid": True, "executable": True, "solution_match": True},
    }
    payload.update(overrides)
    return payload


def qa_contract(formats=("str",)):
    """Build explicit current QA fixture declarations, independent of gold answers."""
    from cypher_3dsg_agent_distillation.cypher.validation_config import ValidationContract

    formats = list(formats)
    shape = formats[0] if formats else "str"
    kind = "set" if shape.startswith("set[") else "list" if shape.startswith("list[") else "scalar"
    value_type = shape[shape.index("[") + 1 : -1] if "[" in shape else shape
    types = {"str": "string", "int": "integer", "float": "number", "point": "point"}
    contract = {
        "result": {"kind": kind, "type": types.get(value_type, "string")},
        "answer": {
            "formats": formats,
            "empty": "<>" if kind == "set" else "[]" if kind == "list" else "",
        },
    }
    return ValidationContract.model_validate(contract).model_dump(mode="json")


def source_metadata(template, bindings=None):
    from cypher_3dsg_agent_distillation.cypher.validation_config import template_identity

    return {"template_identity": template_identity(template), "template_bindings": bindings or {}}
