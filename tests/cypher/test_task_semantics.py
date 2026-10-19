from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from cypher_3dsg_agent_distillation.agents.reasoning_audit import (
    audit_seeds,
    compile_templates,
    load_catalog,
)
from cypher_3dsg_agent_distillation.agents.synthetic_reasoning import _context
from cypher_3dsg_agent_distillation.cypher.task_semantics import (
    build_semantic_context,
    parse_query,
    prepare_seed_semantics,
    resolve_semantics,
    validate_query_semantics,
)
from tests.helpers import qa_contract, source_metadata

CATALOG_PATHS = [str(p) for p in Path("configs/question_templates").glob("*.yaml")]


def test_original_catalog_has_separate_reasoning_config_with_one_global_version():
    catalog = load_catalog(CATALOG_PATHS)
    assert len(catalog) == 91
    assert len({t["query_type"] for t in catalog}) == len(catalog)
    for path in CATALOG_PATHS:
        templates = yaml.safe_load(Path(path).read_text())["templates"]
        assert all(not {"template_id", "oracle_type", "semantics"} & t.keys() for t in templates)
        config = yaml.safe_load(
            (Path("configs/question_templates/synthetic_reasoning") / Path(path).name).read_text()
        )
        assert config["version"] == 1
        assert set(config["templates"]) == {t["query_type"] for t in templates}
        assert all(set(spec) == {"operations"} for spec in config["templates"].values())


def test_bindings_include_nested_coordinates_layers_and_bounds():
    query = (
        'MATCH (s:{case.layer} {nodeSymbol: "{case.symbol}"}) '
        "MATCH (s)-[:{case.relationship}*1..{case.hops}]-(t:{case.layer}) "
        "WHERE point.distance(s.center, point({x: {point.x}, y: {point.y}, "
        "z: {point.z}})) < {radius} RETURN t.nodeSymbol AS target"
    )
    variables = {
        "case": {
            "layer": "MeshPlace",
            "symbol": "P9",
            "relationship": "MESH_PLACE_CONNECTED",
            "hops": 3,
        },
        "point": {"x": -1.5, "y": 2, "z": 0},
        "radius": 5,
    }
    concrete = query
    for field, value in [
        ("case.layer", "MeshPlace"),
        ("case.symbol", "P9"),
        ("case.relationship", "MESH_PLACE_CONNECTED"),
        ("case.hops", 3),
        ("point.x", -1.5),
        ("point.y", 2),
        ("point.z", 0),
        ("radius", 5),
    ]:
        concrete = concrete.replace("{" + field + "}", str(value))
    validate_query_semantics(resolve_semantics(parse_query(query), variables), concrete)


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (o:Object) RETURN unknownFunction(o)",
        "MATCH (o:Object) CALL apoc.foo() RETURN o",
        "MATCH (o:Object)-[:OBJECT_CONNECTED*]-(p:Object) RETURN p",
        "CREATE (o:Object) RETURN o",
    ],
)
def test_unsupported_queries_are_explicit(query):
    with pytest.raises(ValueError, match="Unsupported"):
        parse_query(query)


def test_equivalent_aliases_whitespace_and_conjunction_order_are_accepted():
    original = (
        "MATCH (a:Object), (b:Object) WHERE b <> a AND b.class <> a.class "
        "WITH b, point.distance(a.center, b.center) AS d "
        "ORDER BY d ASC LIMIT 1 RETURN b.nodeSymbol AS object"
    )
    equivalent = (
        "match (reference:Object), (candidate:Object) "
        "where candidate.class <> reference.class AND candidate <> reference "
        "with candidate, point.distance(reference.center, candidate.center) as metric "
        "order by metric limit 1 return candidate.nodeSymbol as object;"
    )
    validate_query_semantics(parse_query(original), equivalent)


@pytest.mark.parametrize(
    "before,after",
    [
        ("distance ASC", "distance DESC"),
        ("< 5", "<= 5"),
        ("t <> s", "t = s"),
        ("*1..2", "*1..3"),
        ("t.nodeSymbol", "t.class"),
    ],
)
def test_meaning_changes_cannot_reuse_an_old_declaration(before, after):
    query = (
        'MATCH (s:Object {nodeSymbol: "O1"}) '
        "MATCH (s)-[:OBJECT_CONNECTED*1..2]-(t:Object) "
        "WHERE t <> s AND point.distance(s.center,t.center) < 5 "
        "WITH t, point.distance(s.center,t.center) AS distance "
        "ORDER BY distance ASC LIMIT 1 RETURN t.nodeSymbol AS object"
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_query_semantics(parse_query(query), query.replace(before, after))


def test_compiler_writes_a_separate_file_and_never_modifies_question_templates(tmp_path):
    path = tmp_path / "templates.yaml"
    destination = tmp_path / "reasoning.yaml"
    original = (
        "# Keep this comment\ntemplates:\n  - query_type: inventory\n"
        "    tags: [example]\n"
        "    cypher: |\n      MATCH (o:Object) RETURN o.class AS class\n"
        "    questions:\n      - List categories.\n"
    )
    path.write_text(original)
    compiled = compile_templates(path, output_path=destination, write=True)
    assert path.read_text() == original
    assert destination.read_text() == compiled
    assert load_fixture_catalog(path, destination)[0]["query_type"] == "inventory"
    assert compile_templates(path) == compiled
    renamed = original.replace("query_type: inventory", "query_type: renamed")
    path.write_text(renamed)
    with pytest.raises(ValueError, match="missing=.*renamed.*obsolete=.*inventory"):
        load_fixture_catalog(path, destination)
    refreshed = compile_templates(path, output_path=destination, write=True)
    assert set(yaml.safe_load(refreshed)["templates"]) == {"renamed"}
    assert path.read_text() == renamed
    assert load_fixture_catalog(path, destination)[0]["query_type"] == "renamed"
    with pytest.raises(ValueError, match="separate from question templates"):
        compile_templates(path, output_path=path, write=True)
    assert path.read_text() == renamed


def test_fingerprint_detects_tampering_and_goal_type_mismatch():
    query = "MATCH (r:Room) RETURN r.nodeSymbol AS room"
    contract = {
        "result": {
            "kind": "role_sets",
            "roles": {"room": {"type": "node_symbol", "layers": ["Room"]}},
        },
        "goal": {"fact": "visited-region", "arguments": [{"result": "room"}]},
    }
    template = {"cypher": query, "validation_contract": contract}
    context = build_semantic_context({}, template, query)
    seed = {"cypher": query, "metadata": {"reasoning_context": deepcopy(context)}}
    seed["metadata"]["reasoning_context"]["goal"]["fact"] = "in-region"
    with pytest.raises(ValueError, match="fingerprint"):
        _context(seed)
    contract["goal"]["fact"] = "visited-object"
    with pytest.raises(ValueError, match="wrong predicate type"):
        build_semantic_context({}, template, query)


def test_audit_counts_coverage_and_cap_omissions_without_changing_seeds():
    query = (
        'MATCH (r:Room)-[:CONTAINS*2..2]->(o:Object) WHERE o.class <> "chair" '
        "RETURN o.nodeSymbol AS object"
    )
    template = {
        "query_type": "contents",
        "cypher": query,
        "validation_contract": qa_contract(["set[str]"]),
    }
    seed = {
        "uid": "one",
        "query_type": "contents",
        "cypher": query,
        "question": "List objects",
        "task_family": "cypher_qa",
        "solution_formats": ["set[str]"],
    }
    seed.update(
        validation_contract=template["validation_contract"], metadata=source_metadata(template)
    )
    original = deepcopy(seed)
    trace = {"uid": "one_teacher", "validation": {"rows": [{"object": "O1"}]}}
    report, migrated = audit_seeds([seed], [template], traces=[trace])
    assert report["verified_seeds"] == 1
    assert report["rules"]["traversal"] == 1
    assert report["caps"]["None"]["turns_with_critical_omissions"] == 0
    assert report["caps"]["1"]["turns_with_critical_omissions"] > 0
    assert seed == original
    assert migrated[0]["metadata"]["reasoning_context"]["retrieval"]


def test_audit_reports_empty_query_reasoning_separately_from_final_syntax():
    query = 'MATCH (o:Object {nodeSymbol: "O1"}) RETURN o.nodeSymbol AS object'
    seed = {
        "uid": "simple",
        "query_type": "lookup",
        "cypher": query,
        "question": "Find O1",
        "task_family": "cypher_qa",
        "solution_formats": ["str"],
    }
    trace = {"uid": "simple_teacher", "validation": {"rows": [{"object": "O1"}]}}
    template = {"query_type": "lookup", "cypher": query, "validation_contract": qa_contract()}
    seed.update(
        validation_contract=template["validation_contract"], metadata=source_metadata(template)
    )
    report, _ = audit_seeds([seed], [template], traces=[trace])
    assert report["renderer_version"] == 3
    assert report["reasoning_by_stage"]["tool"]["turns"] == 1
    assert report["reasoning_by_stage"]["tool"]["turns_with_reasoning"] == 0
    assert report["reasoning_by_stage"]["tool"]["words"]["total"] == 0
    assert report["reasoning_by_stage"]["final"]["turns_with_reasoning"] == 1
    assert report["examples"][0]["final"] == "SLDP unquoted string."
    assert report["caps"]["1"]["turns_with_critical_omissions"] == 0


def test_invalid_function_arity_is_an_explicit_diagnostic():
    with pytest.raises(ValueError, match="argument count"):
        parse_query("MATCH (o:Object) RETURN point.distance(o.center) AS distance")


def test_changed_query_requires_refreshing_only_the_separate_reasoning_file(tmp_path):
    from cypher_3dsg_agent_distillation.agents.synthetic_reasoning import reasoning_sentences

    path = tmp_path / "questions.yaml"
    destination = tmp_path / "reasoning.yaml"
    template = {
        "query_type": "ranked_objects",
        "cypher": "MATCH (o:Object) RETURN o.nodeSymbol AS object ORDER BY object ASC LIMIT 1",
        "questions": ["Choose an object."],
        "tags": ["ordering"],
    }
    path.write_text(yaml.safe_dump({"templates": [template]}))
    compile_templates(path, output_path=destination, write=True)
    template["cypher"] = template["cypher"].replace("ASC", "DESC")
    template["tags"] = ["unrelated"]
    template["questions"] = ["Another wording."]
    path.write_text(yaml.safe_dump({"templates": [template]}))
    question_content = path.read_text()
    with pytest.raises(ValueError, match="does not match"):
        load_fixture_catalog(path, destination)
    compile_templates(path, output_path=destination, write=True)
    catalog = load_fixture_catalog(path, destination)
    seed = {
        "query_type": "ranked_objects",
        "cypher": template["cypher"],
        "validation_contract": catalog[0]["validation_contract"],
        "metadata": source_metadata(template),
    }
    prepare_seed_semantics(seed, catalog[0])
    sentences = reasoning_sentences(seed, stage="tool")
    assert any("DESC" in sentence.text for sentence in sentences)
    assert path.read_text() == question_content


@pytest.mark.parametrize("version", [None, True, "1", 2])
def test_separate_config_rejects_unsupported_global_versions(tmp_path, version):
    path = tmp_path / "questions.yaml"
    destination = tmp_path / "reasoning.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "templates": [
                    {"query_type": "count", "cypher": "MATCH (o:Object) RETURN count(o) AS count"}
                ]
            }
        )
    )
    compiled = yaml.safe_load(compile_templates(path))
    compiled["version"] = version
    destination.write_text(yaml.safe_dump(compiled))
    with pytest.raises(ValueError, match="Unsupported synthetic reasoning version"):
        load_fixture_catalog(path, destination)


def test_real_teachers_and_disabled_synthetic_reasoning_do_not_load_config(monkeypatch):
    from cypher_3dsg_agent_distillation.agents import reasoning_config
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import prepare_synthetic_reasoning

    def forbidden(*args):
        pytest.fail("This teacher must not load synthetic reasoning configuration")

    monkeypatch.setattr(reasoning_config, "load_catalog", forbidden)
    population = [{"query_type": "any_name", "cypher": "arbitrary query", "metadata": {}}]
    original = deepcopy(population)
    question_sets = [{"question_templates_path": "missing_questions.yaml"}]
    for config in [
        {
            "provider": "openrouter",
            "reasoning": {"enabled": True, "semantics_paths": ["missing.yaml"]},
        },
        {"provider": "synthetic", "reasoning": {"enabled": False}},
    ]:
        assert prepare_synthetic_reasoning(population, question_sets, config) == []
        assert population == original


def test_question_generation_retains_bindings_without_compiling_synthetic_semantics(monkeypatch):
    from cypher_3dsg_agent_distillation.cypher import task_semantics
    from cypher_3dsg_agent_distillation.cypher.templates import generate_records

    def forbidden(*args):
        pytest.fail("Question generation must not compile synthetic reasoning semantics")

    monkeypatch.setattr(task_semantics, "parse_query", forbidden)
    records = generate_records(
        {"scene_id": "scene", "values": {"object_classes": ["chair"]}},
        {
            "templates": [
                {
                    "query_type": "custom_inventory",
                    "validation_contract": qa_contract(["set[str]"]),
                    "variables": {"class": "object_classes"},
                    "cypher": 'MATCH (o:Object {class: "{class}"}) RETURN o.nodeSymbol AS object',
                    "questions": ["List {class} objects."],
                    "solution_formats": ["set[str]"],
                }
            ]
        },
        limit_per_template=1,
        warn_on_target_shortfall=False,
    )
    metadata = records[0]["metadata"]
    assert metadata["template_bindings"] == {"class": "chair"}
    assert "reasoning_context" not in metadata


def test_enabled_synthetic_teacher_loads_external_config_and_builds_verified_snapshots():
    from cypher_3dsg_agent_distillation.agents.openrouter_teacher import prepare_synthetic_reasoning

    path = "configs/question_templates/cypher_qa_3dsg.yaml"
    template = next(
        t
        for t in yaml.safe_load(Path(path).read_text())["templates"]
        if t["query_type"] == "object_total"
    )
    from cypher_3dsg_agent_distillation.cypher.validation_config import load_question_catalog

    current = next(
        t for t in load_question_catalog(path)["templates"] if t["query_type"] == "object_total"
    )
    population = [
        {
            "validation_contract": current["validation_contract"],
            "query_type": "object_total",
            "cypher": template["cypher"],
            "metadata": source_metadata(template),
        }
    ]
    paths = prepare_synthetic_reasoning(
        population,
        [{"name": "cypher_qa", "question_templates_path": path}],
        {"provider": "synthetic", "reasoning": {"enabled": True}},
    )
    assert paths[0].name == "cypher_qa_3dsg.yaml"
    assert paths[0].parent.name == "synthetic_reasoning"
    assert paths[0].parent.parent.name == "question_templates"
    assert population[0]["metadata"]["reasoning_context"]["retrieval"]
    assert "reasoning_migration" not in population[0]["metadata"]


def load_fixture_catalog(path, destination):
    from cypher_3dsg_agent_distillation.cypher.validation_config import default_validation_path

    validation = default_validation_path(path)
    validation.parent.mkdir(parents=True, exist_ok=True)
    names = [t["query_type"] for t in yaml.safe_load(path.read_text())["templates"]]
    validation.write_text(yaml.safe_dump({"version": 1, "templates": {name: {} for name in names}}))
    defaults = qa_contract()
    defaults.pop("version")
    (validation.parent / "defaults.yaml").write_text(
        yaml.safe_dump({"version": 1, "defaults": defaults})
    )
    return load_catalog([path], [destination])
