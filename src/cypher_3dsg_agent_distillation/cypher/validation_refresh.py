"""Refresh current record contracts using explicit, verified generation provenance."""

from copy import deepcopy

from cypher_3dsg_agent_distillation.cypher.pddl_goals import render_goal_spec
from cypher_3dsg_agent_distillation.cypher.result_validation import compare_answers
from cypher_3dsg_agent_distillation.cypher.validation_config import (
    ValidationContract,
    contract_for_record,
    fingerprint,
    resolve_contract,
    template_bindings,
)


def refresh_contracts(records, templates):
    catalog = {template["query_type"]: template for template in templates}
    if len(catalog) != len(templates):
        raise ValueError("Contract refresh requires unique template types")
    refreshed = []
    for original in records:
        record = deepcopy(original)
        if "answer_validation" in record:
            raise ValueError(
                "Record-level answer_validation is unsupported; regenerate the dataset"
            )
        previous = contract_for_record(record)
        query_type = record["task_type"] if "messages" in record else record["query_type"]
        try:
            template = catalog[query_type]
        except KeyError as exc:
            raise ValueError(
                f"No selected template for {query_type}; regenerate this dataset"
            ) from exc
        bindings = template_bindings(record, template)
        contract = resolve_contract(template["validation_contract"], bindings)
        for section in ("result", "answer"):
            old = previous.get(section, {}).get("comparison", {})
            new = contract[section]["comparison"]
            if all(old.get(key) == new.get(key) for key in ("mode", "score_source", "limit")):
                if old.get("scores") is not None:
                    new["scores"] = deepcopy(old["scores"])
        ValidationContract.model_validate(contract)
        expected = (
            record.get("expected_solution") if "messages" in record else record.get("solution")
        )
        if expected is not None:
            checked = compare_answers(str(expected), str(expected), contract)
            if checked["status"] == "invalid_expected_solution":
                raise ValueError(f"{record['uid']}: saved label violates the refreshed contract")
        if contract.get("goal"):
            roles = (record.get("grounding") or {}).get("expected")
            if roles is None:
                raise ValueError(f"{record['uid']}: missing expected grounding")
            goal = render_goal_spec(contract["goal"], roles) or ""
            if expected is not None and not compare_answers(goal, str(expected), contract)["match"]:
                raise ValueError(f"{record['uid']}: refreshed goal disagrees with the saved label")
        record["validation_contract"] = contract
        record.setdefault("metadata", {}).update(
            validation_contract_hash=fingerprint(contract),
            validation_provenance=deepcopy(template["validation_provenance"]),
        )
        refreshed.append(record)
    return refreshed
