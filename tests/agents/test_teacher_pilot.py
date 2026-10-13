import io
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cypher_3dsg_agent_distillation.agents.openrouter_teacher import (
    OpenRouterTeacherClient,
    TeacherBudgetExceeded,
)
from cypher_3dsg_agent_distillation.agents.teacher_pilot import (
    TeacherRunJournal,
    pilot_report,
    select_pilot_seeds,
    stratum,
)
from cypher_3dsg_agent_distillation.core.config import load_yaml


def population():
    return [
        {
            "uid": f"{scene}-{task}-{index}",
            "scene_id": scene,
            "task_family": "cypher_qa",
            "query_type": task,
            "tags": ["invalid_input"] if index == 3 else [],
        }
        for scene in ("a", "b")
        for task in ("count", "lookup")
        for index in range(4)
    ]


def response(cost=0.001):
    return {
        "id": "generation",
        "model": "teacher",
        "provider": "provider",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "<answer>3</answer>",
                    "reasoning": "Return a number.",
                },
            }
        ],
        "usage": {
            "cost": cost,
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 15},
        },
    }


def test_sampling_is_reproducible_and_covers_every_task_and_scene():
    first = select_pilot_seeds(population(), extra_per_tag=0)
    second = select_pilot_seeds(list(reversed(population())), extra_per_tag=0)
    assert first == second
    assert len(first) == 4
    assert {(seed["scene_id"], seed["query_type"]) for seed in first} == {
        (scene, task) for scene in ("a", "b") for task in ("count", "lookup")
    }
    assert len(select_pilot_seeds(population(), extra_per_tag=2)) >= 4


def test_journal_resume_replays_paid_response_without_another_api_call(tmp_path):
    sent = []
    journal = TeacherRunJournal(tmp_path)
    client = OpenRouterTeacherClient(
        reasoning={"enabled": True},
        request_sender=lambda request: sent.append(request) or response(),
    )
    client.journal = journal
    client.attempt_id = "seed:0"
    first = client.chat([{"role": "user", "content": "question"}])
    assert sent[0]["reasoning"] == {"enabled": True, "exclude": False}
    assert first["generation_metadata"]["cost_usd"] == 0.001
    assert not journal.pending_path.exists()
    resumed = OpenRouterTeacherClient(
        reasoning={"enabled": True},
        request_sender=lambda request: pytest.fail("duplicate charge"),
    )
    resumed.journal = TeacherRunJournal(tmp_path)
    resumed.attempt_id = "seed:0"
    assert resumed.chat([{"role": "user", "content": "question"}]) == first
    resumed.request_index = 0
    with pytest.raises(ValueError, match="differs"):
        resumed.chat([{"role": "user", "content": "changed"}])


def test_uncertain_interrupted_request_blocks_automatic_rebilling(tmp_path):
    journal = TeacherRunJournal(tmp_path)
    journal.start_request("seed:0", 1, {"model": "teacher"})
    with pytest.raises(RuntimeError, match="may have been billed"):
        TeacherRunJournal(tmp_path)


def test_budget_reserves_next_call_and_stops_on_unknown_billing():
    sent = []
    client = OpenRouterTeacherClient(
        max_cost_usd=0.01,
        max_tokens=100,
        provider={"max_price": {"prompt": 1, "completion": 10}},
        request_sender=lambda request: sent.append(request) or response(0.0095),
    )
    client.chat([{"role": "user", "content": "q"}])
    with pytest.raises(TeacherBudgetExceeded, match="reserves"):
        client.chat([{"role": "user", "content": "q"}])
    assert len(sent) == 1
    client.requests[0]["cost_usd"] = None
    with pytest.raises(TeacherBudgetExceeded, match="unknown cost"):
        client.chat([{"role": "user", "content": "q"}])


def test_weighted_cost_projection_counts_rejects_and_does_not_double_bill_reasoning():
    seeds = population()
    sample = select_pilot_seeds(seeds, extra_per_tag=0)
    attempts = [
        {
            "attempt_id": seed["uid"],
            "seed_uid": seed["uid"],
            "scene_id": seed["scene_id"],
            "task_family": seed["task_family"],
            "query_type": seed["query_type"],
            "stratum": stratum(seed),
            "tags": [],
            "accepted": index != 0,
            "reason": "accepted" if index else "solution_mismatch",
            "reasoning": {"both_stages": True},
            "trainable_with_reasoning": True,
            "matched_trainable": False,
        }
        for index, seed in enumerate(sample)
    ]
    requests = [
        {
            "attempt_id": seed["uid"],
            "cost_usd": 0.01,
            "usage": {
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 15},
            },
        }
        for seed in sample
    ]
    report = pilot_report(seeds, attempts, requests)
    assert report["billing"]["total_cost_usd"] == pytest.approx(0.04)
    assert report["billing"]["completion_tokens"] == 80
    assert report["billing"]["reasoning_tokens"] == 60
    assert report["coverage"]["accepted"] == 3
    assert report["projection"]["full_dataset_cost_usd"] == pytest.approx(0.16)
    assert report["projection"]["expected_usable_reasoning_traces"] == 12
    assert report["projection"]["approximate_95_percent_cost_interval_usd"] is None
    requests[0]["cost_usd"] = None
    report = pilot_report(seeds, attempts, requests)
    assert report["billing"]["total_cost_usd"] is None
    assert report["projection"]["full_dataset_cost_usd"] is None


def test_unsampled_strata_do_not_receive_fabricated_cost_estimates():
    report = pilot_report(population(), [], [])
    assert report["projection"]["full_dataset_cost_usd"] is None
    assert len(report["projection"]["unsampled_strata"]) == 4


@pytest.mark.parametrize(
    "path,model",
    [
        ("configs/teachers/teacher_deepseek_v4_1_flash.yaml", "deepseek/deepseek-v4.1-flash"),
        ("configs/teachers/teacher_glm_5_3_flash.yaml", "z-ai/glm-5.3-flash"),
        ("configs/teachers/teacher_mimo_v2_6_flash.yaml", "xiaomi/mimo-v2.6-flash"),
    ],
)
def test_candidate_configs_request_reasoning_without_synthetic_repair(path, model):
    config = load_yaml(path)
    assert config["model"] == model
    assert config["reasoning"]["enabled"]
    assert "exclude" not in config["reasoning"]
    assert config["routing"]["require_parameters"]
    assert "inject_controlled_repair" not in config


def test_preflight_allows_excluding_mandatory_reasoning_from_dataset():
    model = SimpleNamespace(id="glm", reasoning={"mandatory": True}, supported_parameters=[])
    sdk = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda **kw: SimpleNamespace(result=SimpleNamespace(data=[model]))
        )
    )
    client = OpenRouterTeacherClient(model="glm", sdk_client=sdk, reasoning={"enabled": False})
    client.preflight()
    assert client.api_reasoning() == {"exclude": True}


def test_api_reasoning_blocks_remain_unchanged_in_second_request():
    calls = []
    blocks = [{"type": "reasoning.text", "text": "Use a query.", "signature": "signed"}]

    def send(request):
        calls.append(deepcopy(request))
        result = response()
        result["choices"][0]["message"]["reasoning_details"] = blocks
        return result

    client = OpenRouterTeacherClient(request_sender=send, reasoning={"enabled": True})
    initial = [{"role": "user", "content": "question"}]
    assistant = client.chat(initial)
    client.chat(initial + [assistant, {"role": "user", "content": "continue"}])
    assert calls[-1]["messages"][1]["reasoning_details"] == blocks
    assert "generation_metadata" not in calls[-1]["messages"][1]


def test_native_json_transport_preserves_reasoning_options(monkeypatch):
    from cypher_3dsg_agent_distillation.agents import openrouter_teacher

    sent = []

    def opener(request, *, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps(response()).encode())

    monkeypatch.setattr(openrouter_teacher, "urlopen", opener)
    client = OpenRouterTeacherClient(
        api_key="offline-test",
        reasoning={"enabled": True, "max_tokens": 1000},
    )
    result = client.chat([{"role": "user", "content": "question"}])
    assert sent[0]["reasoning"] == {"enabled": True, "exclude": False, "max_tokens": 1000}
    assert result["generation_metadata"]["cost_usd"] == 0.001
