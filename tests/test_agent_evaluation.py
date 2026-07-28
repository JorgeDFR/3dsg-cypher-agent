from agent_helpers import answerable_trace

from cypher_3dsg_agent_distillation.agent_evaluation import build_agent_summary, evaluate_records


def test_agent_evaluation_counts_tool_and_answer_metrics():
    evaluated = evaluate_records([answerable_trace()])
    summary = build_agent_summary(evaluated)

    assert evaluated[0]["native_tool_call_parse_success"]
    assert evaluated[0]["correct_tool_name"]
    assert evaluated[0]["valid_cypher_argument"]
    assert summary["rates"]["final_answer_match"] == 1.0
    assert summary["average_tool_calls"] == 1.0
