from pathlib import Path

import yaml


def test_project_local_heracles_agent_examples_are_valid_yaml():
    paths = sorted(Path("examples/heracles_agents").rglob("*.yaml"))

    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    native = yaml.safe_load(
        Path("examples/heracles_agents/ollama_distilled_agent_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    fallback = yaml.safe_load(
        Path("examples/heracles_agents/ollama_custom_fallback_agent_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert native["agent_info"]["tool_interface"] == "ollama"
    assert fallback["agent_info"]["tool_interface"] == "custom"
