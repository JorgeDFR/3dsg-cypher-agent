"""Optional full workflow checks using saved or live database execution evidence.

Run with VALIDATION_REPLAY_DATASET pointing to a prepared dataset directory and
VALIDATION_REPLAY_TRACES pointing to a previous teacher_traces.jsonl. No queries
are executed and no model is downloaded. Outputs live in pytest's temporary
directory. Token lengths use a stub; this checks validation and file handoffs,
not the student's tokenizer or a fresh Neo4j import. Set VALIDATION_LIVE_NEO4J_URI
instead of VALIDATION_REPLAY_TRACES for live execution against transaction-local
scene imports; all database changes are rolled back.
"""

import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from cypher_3dsg_agent_distillation.agents import agent_dataset, openrouter_teacher
from cypher_3dsg_agent_distillation.agents.agent_evaluation import recompute_trace_validation
from cypher_3dsg_agent_distillation.agents.agent_records import AgentTraceRecord
from cypher_3dsg_agent_distillation.agents.raw_generation_runtimes import GenerationResult
from cypher_3dsg_agent_distillation.agents.student_evaluation import (
    evaluate_end_to_end_record,
    score_tool_call_generation,
)
from cypher_3dsg_agent_distillation.core import config
from cypher_3dsg_agent_distillation.core.io import read_jsonl
from cypher_3dsg_agent_distillation.core.versioning import atomic_write_yaml


class ReplayTokenizer:
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


class ReplayRuntime:
    def __init__(self, answer, calls):
        def result(text, tool_calls=()):
            return GenerationResult(
                text=text,
                raw_text=text,
                prompt="offline replay",
                prompt_tokens=0,
                generated_tokens=0,
                max_new_tokens=None,
                token_budget=None,
                tool_calls=tool_calls,
            )

        self.responses = iter([result("", calls), result(answer)])

    def generate(self, *args, **kwargs):
        return next(self.responses)


@pytest.fixture
def live_graph():
    uri = os.environ.get("VALIDATION_LIVE_NEO4J_URI")
    if not uri:
        yield None
        return
    from neo4j import GraphDatabase

    from tests.graph.rollback_neo4j import RollbackGraph

    with GraphDatabase.driver(
        uri, auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ["NEO4J_PASSWORD"])
    ) as driver:
        graph = RollbackGraph(driver, os.environ.get("VALIDATION_LIVE_NEO4J_DATABASE", "neo4j"))
        try:
            yield graph
        finally:
            graph.close()


@pytest.mark.skipif(
    not (
        os.environ.get("VALIDATION_REPLAY_DATASET")
        and (
            os.environ.get("VALIDATION_REPLAY_TRACES")
            or os.environ.get("VALIDATION_LIVE_NEO4J_URI")
        )
    ),
    reason="Requires explicitly selected local dataset and saved execution evidence",
)
@pytest.mark.parametrize("reasoning_enabled", [False, True])
def test_complete_dataset_validation_replay(tmp_path, monkeypatch, reasoning_enabled, live_graph):
    import cypher_3dsg_agent_distillation.training.train as training

    source = Path(os.environ["VALIDATION_REPLAY_DATASET"]).resolve()
    executions = {}
    saved = read_jsonl(os.environ["VALIDATION_REPLAY_TRACES"]) if live_graph is None else []
    for record in saved:
        query = None
        for message in record["messages"]:
            for call in message.get("tool_calls") or []:
                function = call.get("function", call)
                if function["name"] == "run_cypher_query":
                    arguments = function["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    query = arguments["cypher_string"].strip()
            if message["role"] == "tool" and query:
                payload = json.loads(message["content"])
                if payload.get("ok"):
                    evidence = {"executable": True, "rows": payload["rows"], "error": None}
                else:
                    error = "\n".join(payload.get("issues") or [])
                    evidence = {
                        "executable": False,
                        "rows": [],
                        "error": error,
                        "syntax_error": "Statement.SyntaxError" in error,
                    }
                executions[(record["scene_id"], query)] = evidence

    destination = tmp_path / source.parent.name / source.name
    destination.mkdir(parents=True)
    for name in ("metadata.yaml", "cypher_seed_records.jsonl", "generated_questions.yaml"):
        shutil.copyfile(source / name, destination / name)
    shutil.copytree(source / "scenes", destination / "scenes")
    seeds = read_jsonl(destination / "cypher_seed_records.jsonl")
    assert seeds and all(seed.get("validation_contract") for seed in seeds)
    monkeypatch.setattr(config, "GENERATED_DATASETS_ROOT", tmp_path)
    current_scene = None
    replayed = []

    class ReplayValidator:
        def __init__(self, **kwargs):
            pass

        def validate(self, cypher, execute=True, **kwargs):
            key = (current_scene, cypher.strip())
            assert key in executions, f"Missing actual saved execution: {key}"
            replayed.append(key)
            return deepcopy(executions[key])

        def close(self):
            pass

    def load_scene(*, scene_path, **kwargs):
        nonlocal current_scene
        current_scene = Path(scene_path).stem

    validator_class = ReplayValidator
    if live_graph is not None:
        syntax_queries = {
            seed["metadata"]["repair_cypher"].strip()
            for seed in seeds
            if (seed.get("metadata") or {}).get("repair_cypher")
        }
        validator_class = live_graph.validator_class(syntax_queries)
        load_scene = live_graph.load_scene
    monkeypatch.setattr(openrouter_teacher, "Neo4jValidator", validator_class)
    monkeypatch.setattr(openrouter_teacher, "load_scene", load_scene)
    monkeypatch.setattr(training, "load_tokenizer", lambda *a: ReplayTokenizer())
    teacher = config.load_yaml("configs/teachers/teacher_synthetic.yaml")
    teacher["reasoning"]["enabled"] = reasoning_enabled
    teacher_path = tmp_path / "teacher.yaml"
    atomic_write_yaml(teacher, teacher_path)
    args = SimpleNamespace(
        teacher_config=str(teacher_path),
        student_config="configs/students/student_qwen3.5_4b.yaml",
        data_config="configs/data.yaml",
        dataset_version=source.name,
        teacher=None,
        teacher_model=None,
        traces_per_seed=None,
        static_only=False,
        max_tool_calls=None,
        preflight=False,
        uri=None,
        username="unused",
        password="unused",
        database="unused",
        neo4j_timeout=10,
        pilot=False,
        pilot_per_task_type=2,
        pilot_extra_per_tag=0,
        sample_seed=42,
        max_cost_usd=None,
        resume=False,
    )
    openrouter_teacher.generate_command(args)
    resolved = config.load_data_config(args.data_config, dataset_version=source.name)
    traces = read_jsonl(resolved["teacher_traces_jsonl_path"])
    rejected = json.loads(Path(resolved["rejected_traces_path"]).read_text())
    assert not rejected, [(r.get("uid"), r.get("reason")) for r in rejected]
    assert len(traces) == len(seeds)
    by_uid = {seed["uid"]: seed for seed in seeds}
    end_to_end_types = set()
    evaluation_scene = None
    for trace in traces:
        current_scene = trace["scene_id"]
        if live_graph is not None and evaluation_scene != current_scene:
            scene = next(s for s in resolved["scenes"] if s["scene_id"] == current_scene)
            live_graph.load_scene(scene_path=scene["scene_path"])
            evaluation_scene = current_scene
        seed = by_uid[trace["metadata"]["seed_uid"]]
        assert trace["validation_contract"] == seed["validation_contract"]
        record = AgentTraceRecord.model_validate(trace)
        if not record.is_refusal:
            recomputed = recompute_trace_validation(record)
            assert recomputed["cypher_solution_match"] is True, record.uid
            assert recomputed["final_answer_match"] is True, record.uid
            assert recomputed["answer_grounded"] is True, record.uid
            assert any(m.get("reasoning") for m in trace["messages"]) == reasoning_enabled
            calls = (
                {
                    "function": {
                        "name": "run_cypher_query",
                        "arguments": {"cypher_string": trace["gold_cypher"]},
                    }
                },
            )
            scored = score_tool_call_generation(
                trace, "", validator=validator_class(), policy=None, structured_tool_calls=calls
            )
            assert scored["status"] == "pass", (record.uid, scored)
            if record.task_type not in end_to_end_types:
                end_to_end_types.add(record.task_type)
                scored = evaluate_end_to_end_record(
                    trace,
                    runtime=ReplayRuntime(record.final_answer_text, calls),
                    validator=validator_class(),
                    policy=None,
                    tool_schemas=[],
                    max_new_tokens=None,
                    enable_thinking=reasoning_enabled,
                    max_tool_rows=10000,
                    max_tool_response_chars=1000000,
                )
                assert scored["status"] == "pass", (record.uid, scored)
                assert scored["final_answer_grounded"] is True, (record.uid, scored)
    assert live_graph is not None or replayed
    agent_dataset.split_command(
        SimpleNamespace(
            data_config=args.data_config,
            dataset_version=source.name,
            split_by_scene=False,
            report_output=None,
            force=False,
            repair_invalid_final_answers=False,
            render_view="ollama_native",
        )
    )
    splits = [r for path in resolved["splits"].values() for r in read_jsonl(path)]
    assert len(splits) == len(traces)
    assert {r["uid"] for r in splits} == {r["uid"] for r in traces}
    assert all(r.get("validation_contract") for r in splits)
