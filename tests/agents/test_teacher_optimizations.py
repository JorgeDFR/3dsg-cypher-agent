import pytest

from cypher_3dsg_agent_distillation.agents.teacher_optimizations import (
    configured_teacher_optimizations,
)


@pytest.mark.parametrize(
    "optimizations",
    [
        [],
        None,
        {"repair_invalid_final_answers": "false"},
        {"recover_earlier_retrievals": "false"},
        {"tuning_prompt": 123},
        {"typo": True},
    ],
)
def test_invalid_optimizations_fail_before_generation(optimizations):
    with pytest.raises(ValueError):
        configured_teacher_optimizations({"provider": "openrouter", "optimizations": optimizations})


def test_synthetic_ignores_enabled_optimizations_with_warnings():
    with pytest.warns(UserWarning) as caught:
        effective = configured_teacher_optimizations(
            {
                "provider": "synthetic",
                "optimizations": {
                    "repair_invalid_final_answers": True,
                    "tuning_prompt": "one call",
                },
            }
        )
    assert len(caught) == 2
    assert effective == {
        "repair_invalid_final_answers": False,
        "recover_earlier_retrievals": False,
        "tuning_prompt": "",
    }


def test_optimizations_default_to_disabled():
    assert configured_teacher_optimizations({"provider": "openrouter"}) == {
        "repair_invalid_final_answers": False,
        "recover_earlier_retrievals": False,
        "tuning_prompt": "",
    }


def test_optimizations_keep_per_teacher_settings():
    assert configured_teacher_optimizations(
        {
            "provider": "openrouter",
            "optimizations": {
                "repair_invalid_final_answers": True,
                "tuning_prompt": " one call ",
            },
        }
    ) == {
        "repair_invalid_final_answers": True,
        "recover_earlier_retrievals": False,
        "tuning_prompt": "one call",
    }


def test_earlier_retrieval_option_is_per_teacher_and_ignored_by_synthetic():
    settings = {"recover_earlier_retrievals": True}
    assert (
        configured_teacher_optimizations(
            {
                "provider": "openrouter",
                "optimizations": settings,
            }
        )["recover_earlier_retrievals"]
        is True
    )
    with pytest.warns(UserWarning, match="recover_earlier_retrievals.*synthetic"):
        assert (
            configured_teacher_optimizations(
                {
                    "provider": "synthetic",
                    "optimizations": settings,
                }
            )["recover_earlier_retrievals"]
            is False
        )
