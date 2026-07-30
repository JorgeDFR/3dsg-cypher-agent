from cypher_3dsg_agent_distillation.experiment_profiles import (
    SUPPORTED_TOOL_INTERFACES,
    load_experiment_profile,
    render_profile_messages,
)


def seed_record(**overrides):
    payload = {
        "uid": "seed",
        "question": "Which object classes exist?",
        "query_type": "object_class_inventory",
        "task_family": "cypher_qa",
        "output_type": "SLDP",
        "gold_cypher": "MATCH (o:Object) RETURN DISTINCT o.class AS class",
        "solution": "<chair, table>",
    }
    payload.update(overrides)
    return payload


def rendered_text(messages):
    return "\n".join(message["content"] for message in messages)


def test_supported_tool_interfaces_are_limited_to_requested_set():
    assert SUPPORTED_TOOL_INTERFACES == ("ollama", "openrouter", "custom")


def test_native_qa_seed_prompt_rendering_excludes_gold_and_solution():
    profile = load_experiment_profile("configs/experiment_profiles/cypher_qa_template.yaml")
    seed = seed_record()

    messages = render_profile_messages(profile, seed, tool_interface="openrouter")
    text = rendered_text(messages)

    assert [message["role"] for message in messages] == ["user"] * len(messages)
    assert text.index("3D scene graph agent") < text.index("<Scene Graph Description>")
    assert text.index("<Scene Graph Description>") < text.index("<Labelspace Description>")
    assert text.index("<Labelspace Description>") < text.index("Labels in Database:")
    assert "Which object classes exist?" in text
    assert "SLDP Equality Language" in text
    assert "Your answer should be an SLDP set" in text
    assert seed["gold_cypher"] not in text
    assert seed["solution"] not in text
    assert "Explain your reasoning" not in text


def test_native_pddl_seed_prompt_rendering_includes_pddl_context_only():
    profile = load_experiment_profile("configs/experiment_profiles/cypher_pddl_template.yaml")
    seed = seed_record(
        question="Grab a wastebin",
        task_family="cypher_pddl",
        output_type="PDDL",
        solution="(or (holding O19) (holding O30))",
    )

    messages = render_profile_messages(profile, seed, tool_interface="custom")
    text = rendered_text(messages)

    assert "PDDL Domain" in text
    assert "PDDL Examples" in text
    assert "PDDL goal language" in text
    assert "Grab a wastebin" in text
    assert seed["solution"] not in text
    assert "Your answer should be an SLDP" not in text
    assert "Explain your reasoning" not in text


def test_profile_rejects_unconfigured_tool_interface():
    profile = load_experiment_profile("configs/experiment_profiles/cypher_qa_template.yaml")
    profile.tool_interfaces = ["ollama"]

    try:
        render_profile_messages(profile, seed_record(), tool_interface="openrouter")
    except ValueError as exc:
        assert "not enabled" in str(exc)
    else:
        raise AssertionError("Expected disabled tool interface error")
