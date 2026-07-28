from pathlib import Path

import yaml


def test_project_local_heracles_agent_examples_are_valid_yaml():
    paths = [
        Path("examples/heracles_agents/ollama_distilled_agent_config.yaml"),
        Path("examples/heracles_agents/ollama_custom_fallback_agent_config.yaml"),
        Path("examples/heracles_agents/model_lists/distilled_students.yaml"),
    ]

    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    native = yaml.safe_load(paths[0].read_text(encoding="utf-8"))
    fallback = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
    assert native["agent_info"]["tool_interface"] == "ollama"
    assert fallback["agent_info"]["tool_interface"] == "custom"

