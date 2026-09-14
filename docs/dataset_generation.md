# Dataset Generation

Dataset construction turns one configured 3DSG scene into validated,
training-ready agent traces. The normal sequence is:

```text
3DSG scene
  -> extracted template values
  -> rendered questions
  -> validated seed records
  -> teacher interactions checked against Neo4j
  -> grouped student splits
```

Commands in this guide use the recommended [Docker stack](../docker/README.md).
For the directory and selection rules of generated artifacts, see
[Artifact versioning](artifact_versioning.md).

## Inputs

`configs/data.yaml` defines the source scene and question sources.

| Key | Meaning |
| --- | --- |
| `scene_id` | Stable identifier used in generated artifact paths |
| `scene_path` | User-provided Hydra/Spark DSG JSON file |
| `schema_path` | 3DSG schema used by template generation and validation |
| `question_sets` | One or more experiment profile and question-template pairs |
| `max_per_task_type` | Optional cap on generated records per task type |
| `split` | Train, evaluation, test, and challenge selection settings |
| `optimize_student_traces` | Whether split rendering prunes repair turns and adds repairable traces |

Multiple task families can share one dataset version:

```yaml
question_sets:
  - name: cypher_qa
    experiment_profile_path: cypher_qa_template.yaml
    question_templates_path: cypher_qa_3dsg.yaml
  - name: cypher_pddl
    experiment_profile_path: cypher_pddl_template.yaml
    question_templates_path: cypher_pddl_3dsg.yaml
```

The legacy singular `experiment_profile_path` and `question_templates_path`
keys are still interpreted as one question set.

## Prepare a dataset version

### 1. Load the scene into Neo4j

Teacher validation executes generated Cypher against the active database.
Reload Neo4j whenever `scene_path` changes:

```bash
docker/stack.sh load-neo4j data/scenes/uhumans2_office.json
docker/stack.sh status
```

### 2. Extract 3DSG values

Extract values that can fill the configured question templates:

```bash
docker/stack.sh main python scripts/extract_3dsg_values.py \
  --data-config configs/data.yaml
```

This command starts a new timestamped dataset version by default. Later
commands select the latest version unless `--dataset-version` is supplied.

### 3. Generate raw questions

Render the question templates from the extracted scene values:

```bash
docker/stack.sh main python scripts/generate_3dsg_seed_templates.py \
  --data-config configs/data.yaml
```

### 4. Prepare seed records

Validate and normalize the generated questions into the internal records used
for teacher generation:

```bash
docker/stack.sh main python scripts/prepare_dataset.py prepare-seeds \
  --data-config configs/data.yaml
```

Seed records contain gold Cypher and expected outputs for validation. They are
internal generation support; the public training and evaluation format is the
agent trace described below.

## Generate teacher traces

### Choose a teacher

| Teacher | Configuration | Behavior | API key |
| --- | --- | --- | --- |
| OpenRouter | `configs/teacher_gpt_oss_120b.yaml` | Generates live tool-using interactions with `openai/gpt-oss-120b` | Required |
| Synthetic | `configs/teacher_synthetic.yaml` | Uses gold Cypher, real Neo4j rows, and the gold final answer | Not required |

For OpenRouter, add `OPENROUTER_API_KEY` to `docker/.env` and verify that the
configured model is available:

```bash
docker/stack.sh main python scripts/generate_traces.py preflight \
  --teacher-config configs/teacher_gpt_oss_120b.yaml
```

Generate and validate OpenRouter traces:

```bash
docker/stack.sh main python scripts/generate_traces.py generate \
  --data-config configs/data.yaml \
  --teacher-config configs/teacher_gpt_oss_120b.yaml
```

Generate deterministic traces without an LLM teacher:

```bash
docker/stack.sh main python scripts/generate_traces.py generate \
  --data-config configs/data.yaml \
  --teacher-config configs/teacher_synthetic.yaml
```

### Teacher validation

The OpenRouter teacher receives the rendered agent prompt and the
`run_cypher_query` schema. It does not receive the seed's gold Cypher or
expected answer. Each interaction is checked through these gates:

1. Enforce static Cypher safety rules.
2. Execute the tool call against the local Neo4j scene.
3. Allow bounded repair attempts (`max_tool_calls` defaults to `3`).
4. Compare QA answers with the expected answer.
5. Verify that PDDL queries satisfy their grounding contract and that the final
   goal is valid for the configured robot domain.

Accepted training traces are written to `teacher_traces.jsonl`. Debug artifacts
include:

| File | Contents |
| --- | --- |
| `accepted.json` | Accepted raw interactions plus validation fields |
| `rejected.json` | Rejected interactions and failure details |
| `teacher_traces.jsonl` | Canonical accepted traces for later splitting |

## Agent trace format

Canonical data is validated as `AgentTraceRecord`:

- Answerable records call `run_cypher_query` and then return a final answer.
- Refusal records make no tool call and return
  `<answer>WRITE_QUERY_NOT_ALLOWED</answer>`.
- `cypher_qa` records use the `SLDP` output type.
- `cypher_pddl` records use the `PDDL` output type and include a grounding
  contract for required result aliases and symbol types.
- Every record identifies its `task_family`, `output_type`, and
  `experiment_profile`.

The supported rendered interfaces are:

- `ollama_native`
- `custom_xml`

## Create student splits

Create the configured grouped splits:

```bash
docker/stack.sh main python scripts/prepare_dataset.py split \
  --data-config configs/data.yaml \
  --render-view ollama_native
```

Splitting is stratified by task family and query type. Task types that share
normalized-equivalent Cypher are assigned together so an equivalent query
cannot leak across splits.

To prune failed repair attempts and include rejected `invalid_answer_format`
traces that are repairable by replacing only the final answer, use:

```bash
docker/stack.sh main python scripts/prepare_dataset.py split \
  --data-config configs/data.yaml \
  --render-view ollama_native \
  --optimize-student-traces
```

This optimization changes only split/render output; the raw teacher traces are
preserved. It can also be enabled with `optimize_student_traces: true` in the
data configuration.

## Select reproducible inputs

Use explicit selectors when a command must not use the latest artifacts:

```bash
docker/stack.sh main python scripts/prepare_dataset.py split \
  --data-config configs/data.yaml \
  --dataset-version 2026-09-08-T-12-05-47-Z \
  --teacher synthetic \
  --teacher-run 2026-09-08-T-12-10-03-Z
```

List valid version identifiers with
[`list_artifact_versions.py`](artifact_versioning.md#inspect-and-select-versions).

## Next steps

- Check generated records with [trace validation](validation.md#trace-validation).
- Fine-tune a model with [Student training](training.md).
