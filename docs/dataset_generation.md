# Dataset Generation

Dataset construction turns one or more configured 3DSG scenes into validated,
training-ready agent traces. The normal sequence is:

```text
3DSG scenes
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

`configs/data.yaml` defines the dataset, its source scenes, and question sources.

| Key | Meaning |
| --- | --- |
| `dataset_id` | Stable identifier used in generated artifact paths |
| `scenes` | Non-empty list of `scene_id` and Hydra/Spark DSG `scene_path` pairs |
| `schema_path` | 3DSG schema used by template generation and validation |
| `question_sets` | One or more experiment profile and question-template pairs |
| `max_per_task_type` | Optional cap on generated records per task type |
| `split` | Train, evaluation, test, and challenge selection settings |

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

A dataset can combine scenes with different layer layouts:

```yaml
dataset_id: uhumans2_spot
scenes:
  - scene_id: uhumans2_office
    scene_path: data/scenes/uhumans2_office.json
  - scene_id: spot_demo
    scene_path: data/scenes/spot_demo.json
```

Question catalogs contain wording, variables, gold queries and sampling settings.
Answer formats, grounding types, PDDL goal trees and comparison policies live in
`configs/question_templates/validation/` and are composed before seed generation.
See [Cypher and grounding validation](cypher_grounding_validation.md) for the schema.

Question templates are generated once per compatible scene. Templates that
require a missing layer are skipped for that scene. For example, room templates
are not generated for a graph without `Room`. Object-containment templates use
a label-agnostic query that accepts either parent layer (`Place` or
`MeshPlace`). Each resulting record keeps its `scene_id`.

`target_questions` is a dataset-wide budget for a question type. Candidates
are divided as evenly as possible among compatible scenes, with unused capacity
redistributed when a scene lacks the layer or values required by that question.
The configured `empty_ratio` is enforced across the combined dataset rather
than independently multiplying the target for every scene. Empty-result and
controlled-repair records are sampled as disjoint shares: an empty question is
never selected for Cypher repair. Generated records use the `invalid_input` tag
when an empty answer comes from an absent explicit reference and the
`controlled_repair` tag for repair questions. Valid count queries whose answer
is zero still count toward the empty quota without being tagged as invalid.

`sampling.max_solution_size` limits the number of entities exposed by a sampled
solution, while `sampling.max_result_rows` also bounds projected Cartesian
expansion for multi-field grounding. Either limit can be overridden on an
individual template. Candidates over a configured limit are skipped before
question selection. Fixed, variable-free inventory and aggregate templates are
left intact.

Sampling promotes diversity at two levels. Within each scene, it first balances
the rendered variables and then uses the expected Spark-DSG solution to break
ties before repeating a wording variant. The dataset-wide scene-selection pass
applies the same variable-first, solution-aware rule while preserving scene and
empty-result quotas. Case providers for constrained questions are derived from
valid graph relationships (for example, place/object-class cases come from
objects actually contained by a place), which avoids spending the budget on
arbitrary combinations that all return the same empty result.

The QA set covers 52 question types. In addition to inventory, containment,
distance, and hop queries, it includes scene totals plus object attribute and
center lookup. It also covers room and place child counts and class projections,
as well as class inclusion, exclusion, and allowed-set constraints. Other queries
include symbol-free class-to-class spatial queries, adjacent-room filtering,
and tied spatial extrema. The PDDL set covers
39 types, including existential and universal class goals, nested conjunctions
and disjunctions, per-object negation, hierarchy-restricted destinations, and
ordered waypoints. Every type provides at least five natural-language variants.

The legacy singular `experiment_profile_path` and `question_templates_path`
keys are still interpreted as one question set.

## Prepare a dataset version

### 1. Start Neo4j

Teacher validation executes generated Cypher against the active database.
Ensure the service is available:

```bash
docker/stack.sh status
```

Trace generation clears and loads the correct configured scene whenever the
next seed has a different `scene_id`. For manual inspection, load one scene by
identifier:

```bash
docker/stack.sh main python scripts/graph/load_3dsg_neo4j.py \
  --data-config configs/data.yaml \
  --scene-id spot_demo
```

### 2. Extract 3DSG values

Extract values that can fill the configured question templates:

```bash
docker/stack.sh main python scripts/graph/extract_3dsg_values.py \
  --data-config configs/data.yaml
```

This command starts a new timestamped dataset version by default. Later
commands select the latest version unless `--dataset-version` is supplied.

### 3. Generate raw questions

Render the question templates from the extracted scene values:

```bash
docker/stack.sh main python scripts/cypher/generate_3dsg_seed_templates.py \
  --data-config configs/data.yaml
```

Each question-template file sets `sampling.repair_cypher_ratio` independently
from `empty_ratio`. The default `0.2` marks 20% of each eligible question type
for controlled repair while keeping the same total question count. Override it
for one generation run with `--repair-cypher-ratio`. Selected records receive a
deterministic, balanced syntax-error mutation in their metadata. Safety-refusal
questions are excluded because their gold value is not executable Cypher.

The available mutations and their coverage across every configured question
type are recorded in [Controlled Cypher repair supervision](cypher_repair_supervision.md).
Regenerate that analysis after changing Cypher templates with:

```bash
python scripts/cypher/analyze_cypher_repair_templates.py
```

### 4. Prepare seed records

Validate and normalize the generated questions into the internal records used
for teacher generation:

```bash
docker/stack.sh main python scripts/agents/prepare_dataset.py prepare-seeds \
  --data-config configs/data.yaml
```

Seed records contain gold Cypher and expected outputs for validation. They are
internal generation support. The public training and evaluation format is the
agent trace described below. Each seed also contains a labelspace prompt built
from the labelspaces embedded in its source scene JSON. Experiment profiles do
not select a scene-specific, hard-coded labelspace file.

## Generate teacher traces

### Choose a teacher

| Teacher | Configuration | Behavior | API key |
| --- | --- | --- | --- |
| OpenRouter | `configs/teachers/teacher_deepseek_v4_1_flash.yaml` | Generates live interactions with DeepSeek V4.1 Flash and reasoning | Required |
| OpenRouter alternatives | `configs/teachers/teacher_glm_5_3_flash.yaml`, `configs/teachers/teacher_mimo_v2_6_flash.yaml` | Cost/intelligence pilot candidates | Required |
| Synthetic | `configs/teachers/teacher_synthetic.yaml` | Uses gold Cypher, real Neo4j rows, and the gold final answer | Not required |

For OpenRouter, add `OPENROUTER_API_KEY` to `docker/.env` and verify that the
configured model is available:

```bash
docker/stack.sh main python scripts/agents/generate_traces.py preflight \
  --teacher-config configs/teachers/teacher_deepseek_v4_1_flash.yaml
```

Teacher-specific optimizations belong in the teacher YAML:

```yaml
optimizations:
  repair_invalid_final_answers: true
  recover_earlier_retrievals: true
  tuning_prompt: >-
    Use at most one tool call per response.
```

All settings default to disabled (an empty string disables `tuning_prompt`).
Synthetic teachers ignore enabled optimizations.
The tuning prompt is appended to the last user message in a copy of every
outbound request, including requests after tool results and repair feedback.
It is excluded from saved conversation messages in training traces and debug
artifacts. Its configured text remains in run metadata for reproducibility,
and changing it prevents resuming a run with a different configuration.
Guidance discourages unwanted behavior; validation and the tool-call limit
still enforce correctness independently.

Generate and validate OpenRouter traces:

```bash
docker/stack.sh main python scripts/agents/generate_traces.py generate \
  --data-config configs/data.yaml \
  --teacher-config configs/teachers/teacher_deepseek_v4_1_flash.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml
```

Generate deterministic traces without an LLM teacher:

```bash
docker/stack.sh main python scripts/agents/generate_traces.py generate \
  --data-config configs/data.yaml \
  --teacher-config configs/teachers/teacher_synthetic.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml
```

Each dataset version contains one `traces/` directory and one `splits/`
directory. Re-running either teacher on a dataset with existing traces or splits
creates a new dataset version, copying prepared inputs such as
`generated_questions.yaml`, `cypher_seed_records.jsonl`, and `scenes/`.
The original traces, splits, and reports are preserved. The new version is
printed in the summary. Split it next, using `--dataset-version` if your data
configuration pins a version. Teacher versions and `--teacher-run` are no longer
used.

To change the scene list or question generation, edit the data configuration
and run `docker/stack.sh demo-dataset`. To change only the teacher, run
`generate_traces.py generate` above against the prepared dataset, then split the
new version.

### Teacher validation

The OpenRouter teacher receives the rendered agent prompt and the
`run_cypher_query` schema. It does not receive the unmodified seed gold Cypher
or expected answer. The invariant instructions from the prompt template use
the chat `system` role. Scene context, examples, and the question use `user`
messages. Each interaction is checked through these gates:

1. Enforce static Cypher safety rules.
2. Execute the tool call against the local Neo4j scene.
3. Allow bounded repair attempts (`max_tool_calls` defaults to `3`).
4. Compare QA answers with the expected answer.
5. Verify that PDDL queries satisfy their grounding contract and that the final
   goal is valid for the configured robot domain.

Tool feedback reports only static safety/schema checks and database execution
errors. Every successfully executed query returns `ok: true` with its rows,
including empty results and exploratory projections. That success flag does
not indicate agreement with a gold solution or grounding contract.

Result decoding, required projections, grounding, and expected-answer comparisons
are internal acceptance checks after the teacher produces its final answer. The
last successfully executed query supplies the retrieval evidence; earlier
exploration is retained but is not combined or selected using gold answers.
Statuses such as `mapping_unresolved`, `invalid_result_shape`, and `mismatch`
appear only in internal/debug reports, never in teacher tool feedback. These
checks still reject incorrect or unresolved completed traces.

The execution-only feedback policy is part of the generation run identity.
Start a fresh generation run for datasets created under the older validation
feedback policy; their journals cannot be resumed under the changed behavior.

For a controlled-repair seed, synthetic trace generation first executes the mutated gold
Cypher and requires Neo4j to return a syntax error. That failed assistant tool
call and its real database error are added before the synthetic corrected call.
Real teachers skip this injection by default and retain their natural corrections.
The synthetic conversation is:

1. Base prompt and question.
2. Mutated Cypher tool call and Neo4j syntax-error response.
3. Corrected Cypher tool call and successful response.
4. Final answer.

Controlled repair therefore requires live Neo4j validation and at least two
allowed tool calls.

Accepted training traces are written to `teacher_traces.jsonl`. Debug artifacts
include:

| File | Contents |
| --- | --- |
| `accepted.json` | Accepted raw interactions plus validation fields |
| `rejected.json` | Rejected interactions and failure details |
| `teacher_traces.jsonl` | Canonical accepted traces for later splitting |
| `token_length_report.yaml` | Per-trace rendered token lengths, p50/p90/p95/p99/max, and the count above the student's `training.max_length` |

The final terminal summary shows the same token-length percentiles and overlength
count. Lengths use the tokenizer and chat template from `--student-config`, so
they match the length check applied during training. Percentiles use the
nearest-rank method.

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
docker/stack.sh main python scripts/agents/prepare_dataset.py split \
  --data-config configs/data.yaml \
  --render-view ollama_native
```

Splitting is stratified by task family and query type. Task types that share
normalized-equivalent seed Cypher are assigned jointly. Grouping uses
`metadata.seed_gold_cypher`, falling back to `gold_cypher` for synthetic or
legacy records, and also connects records sharing an explicit `split_group`.
Generated teacher queries, output aliases, and quote choices do not affect group
identity. The same records follow the same split policy with synthetic and real
teachers. Existing exact per-type count fallbacks still move the minimum
necessary variants when intact groups cannot achieve those counts.

`split.challenge_questions_per_task_type` controls how many records from each
listed `challenge_task_types` entry are generated across all configured scenes,
overriding that template's `target_questions`. Every record from those task
types is assigned to the challenge split, and none appears in train, evaluation,
or test.

Validation-oriented splits are written with all records from one scene
contiguously, minimizing database reloads to once per represented scene. The
training split does not apply scene-specific ordering. The standard SFT sampler
shuffles its records in every training epoch.

Set `split.split_by_scene: true`, or pass `--split-by-scene`, to request
scene-held-out train/eval/test/challenge data. This interface is reserved for a
future implementation and currently exits with
`Scene-aware splitting is not implemented yet.` The default is `false`, so all
configured scenes may contribute to every split.

Enable `optimizations.repair_invalid_final_answers` in the teacher YAML to repair
eligible SLDP `invalid_answer_format` rejections during generation. A separate
`teacher optimizations` progress bar scans completed rejected attempts after
teacher conversations finish (including a clean budget stop). It replaces the
final answer only when saved retrieval evidence verifies the correct result.
Every tool attempt, its feedback, and recorded reasoning are preserved; inline
thinking in the final message is retained in its separate reasoning field.

Enable `optimizations.recover_earlier_retrievals` to recover a correct final
answer grounded in an earlier task retrieval when subsequent calls only inspect
diagnostic information. This preserves the complete conversation, including
opaque reasoning and all diagnostic call/response pairs; it does not rewrite
the final answer. Supported diagnostic suffixes are conservatively identified
from query/schema types and public task constraints:

- Projections that cannot represent the task output, such as inspecting classes
  or counts when the task requires node symbols.
- Inspection of a fixed input node that is not the task's output node.
- Bounded, unfiltered node inventories when the task requires a constrained
  retrieval. An inventory that is itself the task is not skipped.

Selection walks backward past these diagnostics and stops at the first task
query. It never searches older results for a gold match: a newer wrong,
ambiguous, unparsed, or otherwise unsupported task query blocks recovery.
Incomplete, truncated, failed, or mismatched tool feedback also blocks recovery.
The selected retrieval must then pass correctness comparison, and the unchanged
final answer must have valid formatting, be correct, and be grounded in that
retrieval. There must be a completed final answer; exhausted tool budgets are
not recovered. Explicit output adapters and semantic role names remain
authoritative. Queries that cannot be proven diagnostic remain ineligible.

Recovered traces record `retrieval_evidence_policy: earlier_retrievals_v1`, a
one-based `grounding_tool_call_index`, and diagnostic call indices/reasons in
metadata. Saved-trace evaluation and audits recompute the same selection and
check the saved index; they do not blindly trust the marker or cached grade.
Unmarked traces continue to use the latest successful execution. Model-visible
tool responses remain execution-only under either setting.

Repaired and recovered traces are marked in metadata, included in
`teacher_traces.jsonl` and `accepted.json`, and removed from `rejected.json`.
The generation summary shows accepted traces before optimization, final-answer
repairs, earlier retrievals recovered, and final accepted/rejected totals.
`traces/metadata.yaml` and `pilot_report.yaml` record
before/after counts, `repaired` and `recovered` counts, and totals by kind.
Token lengths, coverage, retained UIDs, and cost per accepted trace use the
finalized accepted traces.

The attempt journal keeps the original rejection under
`optimization.raw_rejection`. Optimization commits are atomic; resuming a run
reuses completed attempts, preserves existing optimizations, and does not repeat
paid requests. Both pilot and full generation use this phase. Synthetic teachers
continue to ignore enabled optimizations with a warning.

Splitting only loads the accepted traces finalized during generation; it never
scans or repairs rejected traces. The split flag `--repair-invalid-final-answers`
has been removed. Existing datasets are not changed automatically. The split
report still identifies generation-repaired and recovered records so their
provenance can be reviewed. See [reasoning experiments](reasoning_traces.md) for teacher checks,
the pilot, and reasoning configuration.

## Revalidate saved teacher traces without generation

After fixing validation, regrade a completed run using its saved database rows
and teacher messages. This command never calls the teacher or executes Cypher:

```bash
python scripts/agents/revalidate_dataset.py \
  --dataset-version 2026-10-16-T-12-14-20-Z
```

The default prints proposed grade changes. Add `--apply` to back up existing files
under `revalidation_backups/` and refresh the seeds, questions, canonical traces,
accepted/rejected debug files, attempt journal, token lengths, and coverage report.
Contracts are refreshed only after the saved seed query matches the selected
template. Original answers, grounding labels, and the request/billing journal are
preserved. Incomplete generations and protocol failures remain rejected. Enabled
generation optimizations run against the revalidated evidence. The command uses
the original run's student tokenizer and stores changes in
`reports/revalidation.report.json`; it does not rebuild splits automatically.

Run `prepare_dataset.py split --dataset-version ...` afterwards, followed by
`generate_workflow_report.py --dataset-version ... --dataset-only` to inspect the
updated dataset without training or inference reports.

## Select reproducible inputs

Use explicit selectors when a command must not use the latest artifacts:

```bash
docker/stack.sh main python scripts/agents/prepare_dataset.py split \
  --data-config configs/data.yaml \
  --dataset-version 2026-09-08-T-12-05-47-Z
```

List valid version identifiers with
[`list_artifact_versions.py`](artifact_versioning.md#inspect-and-select-versions).

## Next steps

- Check generated records with [trace validation](validation.md#trace-validation).
- Fine-tune a model with [Student training](training.md).
