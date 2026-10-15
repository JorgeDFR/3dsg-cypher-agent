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


def test_question_wording_cannot_introduce_unqueried_symbols():
    sentences = reasoning_sentences(
        seed(question="Inspect a chair at mesh-place P2 and place p3."), stage="tool"
    )
    text = " ".join(item.text for item in sentences)
    assert "P2" not in text
    assert "p3" not in text
    assert text == ""


def test_label_membership_does_not_claim_scene_instances():
    sentences = reasoning_sentences(
        seed(
            cypher='MATCH (o:Object {class: "unlisted"}) RETURN o',
        ),
        stage="tool",
    )
    assert any("Unlisted class unlisted; query for instances." == item.text for item in sentences)
    assert all("exists in the scene" not in item.text for item in sentences)


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
    query = (
        'MATCH (o:Object {class: "chair"}), (u:Object {class: "rug"}) '
        "RETURN o.nodeSymbol AS visit, u.nodeSymbol AS unheld"
    )
    fields = {field: {"symbol_type": "Object"} for field in ["visit", "unheld"]}
    record = annotated_seed(query, goal, fields)
    assert reasoning_text(record, "tool") == ""
    text = reasoning_text(
        record, "final", rows=[{"visit": "O1", "unheld": "O2"}, {"visit": "O3", "unheld": "O4"}]
    )
    assert "AND(ANY visited-object(chair objects), ALL NOT(holding(rug objects)))" in text
    assert all(symbol not in text for symbol in ["O1", "O2", "O3", "O4"])


def test_repair_has_priority_and_final_answer_covers_empty_and_format():
    message = {"role": "assistant"}
    add_synthetic_reasoning(
        message,
        seed(cypher='MATCH (o:Object {class: "chair"}) WHERE o.nodeSymbol <> "O1" RETURN o'),
        {"enabled": True, "max_sentences": 1},
        stage="tool",
        repair_template="undefined_return_variable",
    )
    assert message["reasoning"] == "Use the bound variable in RETURN."
    assert message["reasoning_selection"]["omitted_rules"]
    message = {"role": "assistant"}
    add_synthetic_reasoning(message, seed(), {"enabled": True}, stage="final", rows=[])
    assert "No grounded targets" in message["reasoning"]
    assert "empty PDDL goal" in message["reasoning"]
    assert "<answer>" not in message["reasoning"]


def test_zero_cap_keeps_auditable_selection_without_reasoning_text():
    message = {"role": "assistant"}
    record = seed(cypher='MATCH (o:Object) WHERE o.class <> "chair" RETURN o')
    add_synthetic_reasoning(message, record, {"enabled": True, "max_sentences": 0}, stage="tool")
    assert "reasoning" not in message
    assert message["reasoning_selection"]["selected_rules"] == []
    assert message["reasoning_selection"]["omitted_rules"]


def annotated_seed(query, goal=None, fields=None, **kwargs):
    template = {"cypher": query, "goal": goal, "grounding": {"fields": fields or {}}}
    return {
        "question": "Any paraphrase",
        "cypher": query,
        "query_type": "arbitrary_name",
        "task_family": "cypher_pddl" if goal else "cypher_qa",
        "solution_formats": ["str"],
        "metadata": {"reasoning_context": semantic_context({}, template, query)},
        **kwargs,
    }


def reasoning_text(record, stage="tool", **kwargs):
    return select_sentences(reasoning_sentences(record, stage=stage, **kwargs), cap=None)[0]


def test_renaming_retagging_and_paraphrasing_do_not_change_reasoning():
    record = annotated_seed(
        'MATCH (s:Object {nodeSymbol: "O11"}), (t:Object) '
        "WHERE t <> s WITH t, point.distance(s.center,t.center) AS distance "
        "ORDER BY distance DESC LIMIT 1 RETURN t.nodeSymbol AS object"
    )
    baseline = reasoning_text(record)
    assert "Farthest" in baseline
    for question, query_type, tags in [
        ("most distant", "renamed", ["nearest"]),
        ("greatest separation", "brand_new_type", []),
        ("closest", "something_else", ["aggregation", "negation"]),
    ]:
        assert (
            reasoning_text({**record, "question": question, "query_type": query_type, "tags": tags})
            == baseline
        )


def test_pretool_explanation_ignores_expected_answers_and_unseen_rows():
    record = annotated_seed('MATCH (o:Object {class: "chair"}) RETURN o.nodeSymbol AS object')
    baseline = reasoning_text(record)
    contaminated = {
        **record,
        "solution": "SECRET_ANSWER",
        "grounding": {"expected": {"object": ["O999"]}, "fields": {}},
        "validation": {"rows": [{"object": "O999"}]},
    }
    assert reasoning_text(contaminated, rows=[{"object": "O999"}]) == baseline
    assert "SECRET_ANSWER" not in baseline and "O999" not in baseline


def test_new_composition_uses_existing_operation_rules():
    query = (
        'MATCH (r:Room {nodeSymbol: "R2"})-[:CONTAINS*2..2]->(o:Object {class: "chair"}) '
        "WITH o, point.distance(o.center,point({x: 0,y: 0,z: 0})) AS distance "
        "WHERE distance <= 5 ORDER BY distance ASC LIMIT 1 RETURN o.nodeSymbol AS object"
    )
    text = reasoning_text(annotated_seed(query))
    assert "CONTAINS 2 hops" in text and "chair objects" in text
    assert "<= 5" in text and "center distance" in text
    assert "Nearest qualifying candidates" in text
    assert text.count("center distance") == 1
    assert "point.distance" not in text


def test_negated_existential_is_one_complete_unit_even_with_cap_one():
    query = 'MATCH (o:Object {class: "computer"}) RETURN o.nodeSymbol AS object'
    goal = {
        "operator": "not",
        "clauses": [{"fact": "holding", "arguments": [{"result": "object"}], "multiple": "or"}],
    }
    record = annotated_seed(query, goal, {"object": {"symbol_type": "Object"}})
    message = {}
    add_synthetic_reasoning(
        message,
        record,
        {"enabled": True, "max_sentences": 1},
        stage="final",
        rows=[{"object": "O1"}, {"object": "O2"}],
    )
    assert message["reasoning"] == "NOT(ANY holding(computer objects))."
    assert message["reasoning_selection"]["selected_rules"] == ["answer:composition"]
    assert message["reasoning_selection"]["omitted_rules"] == ["answer:format"]
    assert message["reasoning_selection"]["critical_omitted_rules"] == []


def test_partial_grounding_follows_nested_goal_structure():
    query = (
        'OPTIONAL MATCH (a:Object {class: "chair"}) '
        'OPTIONAL MATCH (b:Object {class: "desk"}) '
        "RETURN a.nodeSymbol AS first, b.nodeSymbol AS second"
    )
    goal = {
        "operator": "or",
        "clauses": [
            {
                "operator": "and",
                "clauses": [
                    {"fact": "safe", "arguments": [{"result": "first"}]},
                    {"fact": "visited-object", "arguments": [{"result": "second"}]},
                ],
            },
            {"fact": "at-object", "arguments": [{"result": "second"}]},
        ],
    }
    fields = {name: {"symbol_type": "Object"} for name in ["first", "second"]}
    rows = [{"first": None, "second": "O7"}]
    record = annotated_seed(query, goal, fields)
    text = reasoning_text(record, "final", rows=rows)
    assert "Drop ungrounded alternatives" in text
    assert "at-object" in text and "O7" not in text
    assert "safe" not in text
    required = annotated_seed(query, goal["clauses"][0], fields)
    assert "Required goal branch unavailable" in reasoning_text(required, "final", rows=rows)
    assert "empty PDDL goal" in reasoning_text(required, "final", rows=rows)


@pytest.mark.parametrize(
    "shape,empty",
    [
        ("set[str]", "Empty SLDP set: <>."),
        ("list[str]", "Empty SLDP list: []."),
        ("str", "Empty SLDP string."),
    ],
)
def test_empty_answers_use_the_declared_container(shape, empty):
    record = annotated_seed(
        "MATCH (o:Object) RETURN o.nodeSymbol AS object", solution_formats=[shape]
    )
    assert empty == reasoning_text(record, "final", rows=[])


def test_count_zero_is_not_an_empty_scalar_or_collection():
    record = annotated_seed("MATCH (o:Object) RETURN count(o) AS count", solution_formats=["int"])
    assert "SLDP integer." == reasoning_text(record, "final", rows=[{"count": 0}])
    assert "Empty" not in reasoning_text(record, "final", rows=[{"count": 0}])


def test_pair_projection_preserves_rows_and_all_ties():
    from cypher_3dsg_agent_distillation.agents.reasoning_audit import load_catalog

    template = next(
        t
        for t in load_catalog(["configs/question_templates/cypher_qa_3dsg.yaml"])
        if t["query_type"] == "closest_room_pairs"
    )
    record = annotated_seed(template["cypher"], solution_formats=["set[list[str]]"])
    tool = reasoning_text(record)
    assert "Unordered pairs once" in tool
    assert "Keep all minimum" in tool
    assert tool.count("center distance") == 1
    final = reasoning_text(record, "final", rows=[{"first": "R1", "second": "R2"}])
    assert "preserve row pairs" in final
    assert "R1" not in final and "R2" not in final


def test_selection_priority_and_prose_order_are_independent_and_scope_is_retained():
    candidates = [
        Sentence("essential", 10, "Explain the task.", order=10, critical=True),
        Sentence("identity", 40, "Match the reference.", order=0),
        Sentence("same-text-other-scope", 40, "Match the reference.", order=0, scope="alternative"),
    ]
    text, diagnostic = select_sentences(candidates, cap=None)
    assert text.startswith("Match the reference. Match the reference.")
    assert diagnostic["candidate_count"] == 3
    limited, diagnostic = select_sentences(candidates, cap=1)
    assert limited == "Explain the task."
    assert diagnostic["critical_omitted_rules"] == []


def test_null_projection_is_empty_but_numeric_zero_is_retained():
    record = annotated_seed("OPTIONAL MATCH (o:Object) RETURN o.nodeSymbol AS object")
    assert "Empty SLDP string." == reasoning_text(record, "final", rows=[{"object": None}])
    assert "Empty" not in reasoning_text(record, "final", rows=[{"object": 0}])


def test_singleton_negation_does_not_invent_a_disjunction():
    query = 'MATCH (o:Object {nodeSymbol: "O1"}) RETURN o.nodeSymbol AS object'
    goal = {
        "operator": "not",
        "clauses": [{"fact": "holding", "arguments": [{"result": "object"}], "multiple": "or"}],
    }
    record = annotated_seed(query, goal, {"object": {"symbol_type": "Object"}})
    text = reasoning_text(record, "final", rows=[{"object": "O1"}])
    assert "NOT(holding(object))." in text
    assert "ANY" not in text and "OR" not in text


def test_repair_identifies_the_actual_function_from_the_faulty_query():
    query = "MATCH (o:Object) RETURN count(o) AS count"
    record = annotated_seed(query)
    record["metadata"]["repair_cypher"] = query.replace("count(", "unknownCypherFunction(")
    assert "Replace unknownCypherFunction with count." in reasoning_text(
        record, repair_template="unknown_function"
    )


def test_final_reasoning_omits_columns_values_and_answer_tag_reminders():
    query = "MATCH (o:Object) WITH o.class AS class RETURN class"
    record = annotated_seed(query, solution_formats=["set[str]"])
    text = reasoning_text(record, "final", rows=[{"class": "chair"}])
    assert "SLDP set of unique values." == text
    assert "class" not in text and "chair" not in text
    assert "answer" not in text


def test_unknown_result_alias_does_not_invent_an_empty_outcome():
    record = annotated_seed("MATCH (o:Object) RETURN o.nodeSymbol AS object")
    text = reasoning_text(record, "final", rows=[{"renamed": "O1"}])
    assert "SLDP unquoted string." == text


def test_query_reasoning_is_independent_of_pddl_actions_and_goal_logic():
    query = (
        'MATCH (r:Room {nodeSymbol: "R2"})-[:CONTAINS*2..2]->(o:Object {class: "chair"}) '
        "RETURN o.nodeSymbol AS object"
    )
    fields = {"object": {"symbol_type": "Object"}}
    visit = {"fact": "visited-object", "arguments": [{"result": "object"}], "multiple": "or"}
    inspect = {"fact": "safe", "arguments": [{"result": "object"}], "multiple": "and"}
    no_holding = {
        "operator": "not",
        "clauses": [{"fact": "holding", "arguments": [{"result": "object"}], "multiple": "or"}],
    }
    goals = [visit, inspect, no_holding, {"operator": "and", "clauses": [visit, no_holding]}]
    records = [annotated_seed(query, goal, fields) for goal in goals]
    tool_texts = {reasoning_text(record) for record in records}
    assert len(tool_texts) == 1
    tool = tool_texts.pop()
    assert "CONTAINS 2 hops" in tool
    assert all(word not in tool for word in ["visited-object", "holding", "safe", "ANY", "ALL"])
    rows = [{"object": "O1"}, {"object": "O2"}]
    assert len({reasoning_text(record, "final", rows=rows) for record in records}) == len(goals)


def test_outer_negation_and_per_target_negation_remain_distinct():
    query = 'MATCH (o:Object {class: "computer"}) RETURN o.nodeSymbol AS object'
    fields = {"object": {"symbol_type": "Object"}}
    atom = {"fact": "holding", "arguments": [{"result": "object"}], "multiple": "or"}
    rows = [{"object": "O1"}, {"object": "O2"}]
    outside = annotated_seed(query, {"operator": "not", "clauses": [atom]}, fields)
    inside = annotated_seed(query, {**atom, "multiple": "and", "negate_each": True}, fields)
    assert "NOT(ANY holding(computer objects))." in reasoning_text(outside, "final", rows=rows)
    assert "ALL NOT(holding(computer objects))." in reasoning_text(inside, "final", rows=rows)


def test_complete_compound_alternative_survives_one_unit_cap():
    query = (
        'MATCH (a:Object {class: "chair"}), (b:Object {class: "desk"}) '
        "RETURN a.nodeSymbol AS first, b.nodeSymbol AS second"
    )
    goal = {
        "operator": "or",
        "clauses": [
            {
                "operator": "and",
                "clauses": [
                    {"fact": "safe", "arguments": [{"result": "first"}], "multiple": "or"},
                    {
                        "fact": "visited-object",
                        "arguments": [{"result": "second"}],
                        "multiple": "or",
                    },
                ],
            },
            {"fact": "at-object", "arguments": [{"result": "second"}], "multiple": "or"},
        ],
    }
    fields = {field: {"symbol_type": "Object"} for field in ["first", "second"]}
    record = annotated_seed(query, goal, fields)
    message = {"role": "assistant"}
    add_synthetic_reasoning(
        message,
        record,
        {"enabled": True, "max_final_sentences": 1},
        stage="final",
        rows=[{"first": "O1", "second": "O3"}, {"first": "O2", "second": "O4"}],
    )
    assert message["reasoning"] == (
        "OR(AND(ANY safe(chair objects), ANY visited-object(desk objects)), "
        "ANY at-object(desk objects))."
    )
    assert message["reasoning_selection"]["critical_omitted_rules"] == []


def test_placement_preserves_object_destination_argument_order_without_symbols():
    query = (
        'MATCH (o:Object {nodeSymbol: "O1"}), (p:Place {nodeSymbol: "p2"}) '
        "RETURN o.nodeSymbol AS object, p.nodeSymbol AS destination"
    )
    goal = {
        "fact": "object-in-place",
        "arguments": [{"result": "object"}, {"result": "destination"}],
    }
    fields = {"object": {"symbol_type": "Object"}, "destination": {"symbol_type": "Place"}}
    record = annotated_seed(query, goal, fields)
    text = reasoning_text(record, "final", rows=[{"object": "O1", "destination": "p2"}])
    assert "object-in-place(object, place)." in text
    assert "O1" not in text and "p2" not in text


def test_equal_class_targets_keep_distinct_roles():
    query = (
        'MATCH (reference:Object {class: "chair"}), (target:Object {class: "chair"}) '
        "WHERE target <> reference AND point.distance(reference.center, target.center) < 5 "
        "RETURN reference.nodeSymbol AS first, target.nodeSymbol AS second"
    )
    goal = {
        "operator": "and",
        "clauses": [
            {"fact": "visited-object", "arguments": [{"result": "first"}]},
            {"fact": "at-object", "arguments": [{"result": "second"}]},
        ],
    }
    fields = {field: {"symbol_type": "Object"} for field in ["first", "second"]}
    record = annotated_seed(query, goal, fields)
    tool = reasoning_text(record)
    assert "reference chair objects" in tool and "candidate chair objects" in tool
    text = reasoning_text(record, "final", rows=[{"first": "O1", "second": "O2"}])
    assert "visited-object(reference chair objects)" in text
    assert "at-object(candidate chair objects)" in text


@pytest.mark.parametrize(
    "shape,required",
    [
        ("int", "SLDP integer"),
        ("str", "unquoted string"),
        ("point", "POINT(x y z); no commas"),
        ("set[point]", "unique POINT(x y z); no coordinate commas"),
        ("set[str]", "set of unique values"),
        ("list[str]", "list; preserve order"),
        ("set[list[str]]", "symbol pairs; preserve row pairs"),
    ],
)
def test_routine_syntax_remains_without_result_narration(shape, required):
    record = annotated_seed(
        "MATCH (o:Object) RETURN o.nodeSymbol AS object", solution_formats=[shape]
    )
    text = reasoning_text(record, "final", rows=[{"object": "O999"}])
    assert required in text
    assert "O999" not in text and "column" not in text and "answer" not in text


def test_repeated_unlisted_class_notice_is_deduplicated():
    record = annotated_seed(
        'MATCH (a:Object {class: "unknown"}), (b:Object {class: "unknown"}) '
        "RETURN a.nodeSymbol AS first, b.nodeSymbol AS second",
        labelspace_description="<object_labels>chair</object_labels>",
    )
    assert reasoning_text(record) == "Unlisted class unknown; query for instances."


def test_optional_candidate_retrieval_does_not_describe_pddl_action():
    record = annotated_seed(
        'OPTIONAL MATCH (a:Object {class: "chair"}) RETURN a.nodeSymbol AS object'
    )
    text = reasoning_text(record)
    assert "Allow missing chair objects." == text


def test_aggregation_ranking_and_tied_extrema_avoid_repeated_computations():
    from cypher_3dsg_agent_distillation.agents.reasoning_config import load_catalog

    catalog = load_catalog(["configs/question_templates/cypher_qa_3dsg.yaml"])
    query = next(t["cypher"] for t in catalog if t["query_type"] == "most_connected_rooms")
    candidates = reasoning_sentences(annotated_seed(query), stage="tool")
    text, _ = select_sentences(candidates, cap=None)
    assert "Keep all maximum" in text and "per rooms" in text
    assert text.count("count(") == 1
    assert not any(
        c.rule.startswith(("aggregation:", "extremum:", "distance:")) for c in candidates
    )


def test_arbitrary_integer_comparisons_are_not_labeled_pair_deduplication():
    record = annotated_seed(
        "MATCH (a:Object), (b:Object) WHERE toInteger(a.class) < toInteger(b.class) "
        "RETURN a.nodeSymbol AS object"
    )
    text = reasoning_text(record)
    assert "<" in text
    assert "Unordered pairs" not in text
