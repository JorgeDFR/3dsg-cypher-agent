from cypher_3dsg_agent_distillation.agents.experiment_profiles import (
    SUPPORTED_TOOL_INTERFACES,
    experiment_profile_for_seed,
    load_experiment_profile,
    load_experiment_profiles_from_config,
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
        "solution_formats": ["set[str]"],
        "labelspace_description": (
            "<Labelspace Description>\n"
            "Labels in Database:\n"
            "<object_labels>\nchair\ntable\n</object_labels>\n"
            "</Labelspace Description>"
        ),
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
    assert text.index(
        "mapping from natural language queries to Cypher queries"
    ) < text.index("<Scene Graph Description>")
    assert text.index("<Scene Graph Description>") < text.index("<Labelspace Description>")
    assert text.index("<Labelspace Description>") < text.index("Labels in Database:")
    assert "Which object classes exist?" in text
    assert "## SLDP Language" in text
    assert "return the appropriate empty value" in text
    assert "Your answer should be an SLDP set of strings" in text
    assert seed["gold_cypher"] not in text
    assert seed["solution"] not in text
    assert "Explain your reasoning" not in text


def test_seed_labelspace_is_used_instead_of_a_profile_specific_file():
    profile = load_experiment_profile("configs/experiment_profiles/cypher_qa_template.yaml")
    messages = render_profile_messages(
        profile,
        seed_record(labelspace_description="SCENE-SPECIFIC-LABELSPACE"),
        tool_interface="openrouter",
    )

    assert "SCENE-SPECIFIC-LABELSPACE" in rendered_text(messages)
    assert profile.labelspace_path is None


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
    assert "## PDDL Goal Language" in text
    assert "return an empty answer instead of inventing a PDDL goal" in text
    assert text.index("<PDDL Domain>") < text.index("<PDDL Examples>")
    assert text.index("<PDDL Examples>") < text.index(
        "Now, use Cypher queries to generate a PDDL goal"
    )
    assert "Grab a wastebin" in text
    assert seed["solution"] not in text
    assert "Your answer should be an SLDP" not in text
    assert "Explain your reasoning" not in text


def test_native_qa_seed_prompt_rendering_uses_solution_formats_for_hint():
    profile = load_experiment_profile("configs/experiment_profiles/cypher_qa_template.yaml")

    point_messages = render_profile_messages(
        profile,
        seed_record(
            query_type="object_class_centers",
            solution="<POINT(1 2 3)>",
            solution_formats=["set[point]"],
        ),
        tool_interface="openrouter",
    )
    count_messages = render_profile_messages(
        profile,
        seed_record(
            query_type="object_class_count",
            solution=3,
            solution_formats=["int"],
        ),
        tool_interface="openrouter",
    )
    scalar_or_set_messages = render_profile_messages(
        profile,
        seed_record(
            query_type="objects_in_bounding_box",
            solution="O1",
            solution_formats=["set[str]", "str"],
        ),
        tool_interface="openrouter",
    )

    assert "Your answer should be an SLDP set of points" in rendered_text(point_messages)
    assert "Your answer should be an SLDP primitive number" in rendered_text(count_messages)
    assert (
        "Your answer should be an SLDP set of strings or a primitive string"
        in rendered_text(scalar_or_set_messages)
    )


def test_profile_rejects_unconfigured_tool_interface():
    profile = load_experiment_profile("configs/experiment_profiles/cypher_qa_template.yaml")
    profile.tool_interfaces = ["ollama"]

    try:
        render_profile_messages(profile, seed_record(), tool_interface="openrouter")
    except ValueError as exc:
        assert "not enabled" in str(exc)
    else:
        raise AssertionError("Expected disabled tool interface error")


def test_mixed_config_resolves_profile_per_seed():
    profiles = load_experiment_profiles_from_config(
        {
            "scene_id": "scene",
            "scene_path": "scene.json",
            "question_sets": [
                {
                    "name": "qa",
                    "question_templates_path": "configs/question_templates/cypher_qa_3dsg.yaml",
                    "experiment_profile_path": (
                        "configs/experiment_profiles/cypher_qa_template.yaml"
                    ),
                },
                {
                    "name": "pddl",
                    "question_templates_path": "configs/question_templates/cypher_pddl_3dsg.yaml",
                    "experiment_profile_path": (
                        "configs/experiment_profiles/cypher_pddl_template.yaml"
                    ),
                },
            ],
        }
    )

    profile = experiment_profile_for_seed(
        {
            "uid": "pddl",
            "task_family": "cypher_pddl",
            "output_type": "PDDL",
            "experiment_profile": "cypher_pddl_template",
        },
        profiles,
    )

    assert profile.name == "cypher_pddl_template"


def test_multiple_question_sets_may_share_one_profile():
    profiles = load_experiment_profiles_from_config(
        {
            "scene_id": "scene",
            "scene_path": "scene.json",
            "question_sets": [
                {
                    "name": name,
                    "question_templates_path": f"{name}.yaml",
                    "experiment_profile_path": (
                        "configs/experiment_profiles/cypher_qa_template.yaml"
                    ),
                }
                for name in ("qa_core", "qa_spatial")
            ],
        }
    )

    assert list(profiles) == ["cypher_qa_template"]
