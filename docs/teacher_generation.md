# Teacher Generation

Generate validated tool interactions from the prepared seeds in
[Dataset generation](dataset_generation.md#prepare-a-dataset-version).
Commands use the [Docker stack](../docker/README.md) with Neo4j running.
Generation replaces the active database scene as it moves between inputs.

## Choose a teacher

| Teacher | Configuration | Requirements |
| --- | --- | --- |
| Synthetic | `configs/teachers/teacher_synthetic.yaml` | Neo4j, no API key |
| OpenRouter | `configs/teachers/teacher_deepseek_v4_1_flash.yaml` | Neo4j and `OPENROUTER_API_KEY` |
| Other OpenRouter profiles | `teacher_glm_5_3_flash.yaml`, `teacher_mimo_v2_6_flash.yaml` in `configs/teachers/` | Same as OpenRouter |

Synthetic generation uses reference queries and real database results.
OpenRouter generates live conversations. Set its API key in `docker/.env`.
Use [Teacher checks and pilots](teacher_checks.md) to check availability,
reasoning coverage, cost, and acceptance before a full remote run.

## Generate traces

Generate deterministic traces without a remote model:

```bash
docker/stack.sh main python scripts/agents/generate_traces.py generate \
  --data-config configs/data.yaml \
  --dataset-version DATASET_VERSION \
  --teacher-config configs/teachers/teacher_synthetic.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml
```

For remote generation, replace `--teacher-config` with an OpenRouter profile.
The student configuration selects the tokenizer used to check trace lengths.

Generation prints the output dataset version. An existing trace set or split
causes generation to fork a new version with the prepared inputs. See
[Artifact selection rules](artifact_versioning.md#selection-rules).

After generation, [create student splits](dataset_generation.md#create-student-splits)
using the printed version. To change scenes or questions, prepare a new dataset.
To change only the teacher, reuse the prepared inputs.

## Generation settings

Set these in the teacher YAML:

| Setting | Purpose |
| --- | --- |
| `traces_per_seed` | Number of interactions generated for each seed |
| `max_tool_calls` | Maximum tool calls per interaction, default `3` |
| `max_tokens` | Response token limit, including reasoning where the provider counts it |
| `reasoning` | Reasoning inclusion and provider-specific limits |
| `optimizations` | Optional answer repair, retrieval recovery, and request guidance |

See [Reasoning experiments](reasoning.md) for reasoning settings and matched
training comparisons. Remote generation can use spending limits and resumable
journals as described in [Teacher checks and pilots](teacher_checks.md#resume-a-stopped-run).

## Optional optimizations

```yaml
optimizations:
  repair_invalid_final_answers: true
  recover_earlier_retrievals: true
  tuning_prompt: >-
    Use at most one tool call per response.
```

| Option | Effect |
| --- | --- |
| `repair_invalid_final_answers` | Repair eligible SLDP formatting failures when saved query results verify the answer |
| `recover_earlier_retrievals` | Accept an earlier complete retrieval when later calls only inspect diagnostic information and the final answer is correct and grounded |
| `tuning_prompt` | Add teacher-specific guidance to each model request |

These settings default to disabled. Synthetic teachers ignore them.
Recovery preserves the conversation and final answer. A newer conflicting or
unresolved task retrieval blocks recovery.

The tuning prompt is excluded from saved training messages but retained in run
metadata. Changing it prevents resuming the same run. It does not override
validation or tool-call limits.

The generation summary and pilot report show repair and recovery counts.
Splitting uses the finalized accepted traces.

## Controlled Cypher repair

The synthetic teacher can inject a faulty query before its corrected call.
Enable `inject_controlled_repair` and allow at least two tool calls. Neo4j must
confirm that the faulty query produces a syntax error.

Each repair example contains the faulty call, real error feedback, corrected
call, and final answer. The faulty call and error are context-only during
training. Remote teachers retain their natural errors and corrections without
injected failures.

To inspect mutation coverage and observed syntax failures:

```bash
docker/stack.sh main python scripts/cypher/analyze_cypher_repair_templates.py
```

The generated report is saved to
`outputs/reports/cypher_repair_supervision.md`. Use `--output PATH` for another
location.

## Acceptance and outputs

A training trace must pass query safety, execution, retrieval correctness,
answer formatting, and final-answer correctness and grounding checks.
Refusal questions return `<answer>WRITE_QUERY_NOT_ALLOWED</answer>` without a
tool call. QA answers use SLDP, while planning answers use PDDL goals.

Remote teachers receive the question, scene context, and tool schema.
Tool feedback contains database rows or safety/execution errors. Correctness
and grounding are checked separately after the final answer. See
[Validation contracts](validation_contracts.md) for the result rules.

| File under `traces/` | Contents |
| --- | --- |
| `teacher_traces.jsonl` | Accepted traces used for splitting |
| `accepted.json` | Accepted interactions and validation details |
| `rejected.json` | Rejected interactions and failure reasons |
| `token_length_report.yaml` | Length percentiles and counts above `training.max_length` |

Inspect outcomes in the [HTML workflow report](workflow_report.md). To update
grades from saved evidence, use [Dataset audits](dataset_audits.md).
