import pytest

from cypher_3dsg_agent_distillation.agents.synthetic_reasoning import (
    Sentence,
    add_synthetic_reasoning,
    reasoning_sentences,
    select_sentences,
    semantic_context,
    sentence_cap,
)


def seed(**kwargs):
    return {
        "question": "Inspect any chair, then finish in any office.",
        "task_family": "cypher_pddl",
        "cypher": 'MATCH (o:Object {class: "chair"}), (r:Room {class: "office"}) RETURN o,r',
        "labelspace_description": "<object_labels>chair\nbox</object_labels>"
        "<room_labels>office</room_labels>",
        **kwargs,
    }


def test_unlimited_default_and_deterministic_capping_record_omitted_rules():
    candidates = [
        Sentence("format", 5, "Use answer tags."),
        Sentence("outcome", 0, "No rows were returned."),
        Sentence("extra", 10, "Be concise."),
        Sentence("duplicate", 20, "Be concise."),
    ]
    text, report = select_sentences(candidates, cap=None)
    assert text == "No rows were returned. Use answer tags. Be concise."
    assert report["candidate_count"] == 3
    assert report["omitted_rules"] == []
    text, report = select_sentences(list(reversed(candidates)), cap=2)
    assert text == "No rows were returned. Use answer tags."
    assert report["selected_rules"] == ["outcome", "format"]
    assert report["omitted_rules"] == ["extra"]
    assert sentence_cap({}, "tool") is None
    assert sentence_cap({"max_sentences": 2, "max_final_sentences": None}, "final") is None


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_invalid_caps_rejected(value):
    with pytest.raises(ValueError, match="caps"):
        sentence_cap({"max_sentences": value}, "tool")


def test_only_relevant_class_and_symbol_rules_are_selected():
    sentences = reasoning_sentences(
        seed(question="Inspect a chair at mesh-place P2 and place p3."), stage="tool"
    )
    text = " ".join(item.text for item in sentences)
    assert "P2 identifies MeshPlace" in text
    assert "p3 identifies Place" in text
    assert "chair is in the object label space" in text
    assert "office is in the room label space" in text
    assert "No grounding" not in text
    assert "Rank by" not in text


def test_label_membership_does_not_claim_scene_instances():
    sentences = reasoning_sentences(
        seed(
            cypher='MATCH (o:Object {class: "unlisted"}) RETURN o',
        ),
        stage="tool",
    )
    assert any("unlisted is not in the object label space" in item.text for item in sentences)


def test_nested_pddl_goal_scope_and_action_mapping_survive_semantic_annotation():
    goal = {
        "operator": "and",
        "clauses": [
            {"fact": "visited-object", "arguments": [{"result": "visit"}], "multiple": "or"},
            {
                "fact": "holding",
                "arguments": [{"result": "unheld"}],
                "multiple": "and",
                "negate_each": True,
            },
        ],
    }
    context = semantic_context({"class": "chair"}, {"goal": goal}, seed()["cypher"])
    sentences = reasoning_sentences(seed(metadata={"reasoning_context": context}), stage="tool")
    text = " ".join(item.text for item in sentences)
    assert "or for any matching target in the visit condition" in text
    assert "and for every matching target in the unheld condition" in text
    assert "Negate each" in text
    assert "visited-object" in text
    assert context["variables"] == {"class": "chair"}


def test_repair_has_priority_and_final_answer_covers_empty_and_format():
    message = {"role": "assistant"}
    add_synthetic_reasoning(
        message,
        seed(),
        {"enabled": True, "max_sentences": 1},
        stage="tool",
        repair_template="undefined_return_variable",
    )
    assert message["reasoning"] == "Use the variable bound by MATCH in RETURN."
    assert message["reasoning_selection"]["omitted_rules"]
    message = {"role": "assistant"}
    add_synthetic_reasoning(message, seed(), {"enabled": True}, stage="final", rows=[])
    assert "No grounding" in message["reasoning"]
    assert "PDDL goal inside answer tags" in message["reasoning"]


def test_zero_cap_keeps_auditable_selection_without_reasoning_text():
    message = {"role": "assistant"}
    add_synthetic_reasoning(message, seed(), {"enabled": True, "max_sentences": 0}, stage="tool")
    assert "reasoning" not in message
    assert message["reasoning_selection"]["selected_rules"] == []
    assert message["reasoning_selection"]["omitted_rules"]
