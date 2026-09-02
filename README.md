# 3DSG Cypher Agent Distillation

This repository fine-tunes and distills small tool-using agents for querying
Neo4j-backed 3D Scene Graphs. A dataset can mix direct question answering and
robot-planning instructions: both call
`run_cypher_query(cypher_string: str)`, then return either an SLDP answer or a
grounded PDDL goal. Tool calls use either Ollama-native calls or a custom XML
fallback interface.

- Teacher options: `openai/gpt-oss-120b` through OpenRouter, or a deterministic
  `synthetic` teacher using gold Cypher and live Neo4j results
- Primary student: `qwen-3.5-4b` through Hugging Face, and exported to Ollama

## Scope

The active workflow is:

1. Generate or load validated 3DSG Cypher seed records from one or more question sets.
2. Generate tool-using traces with either an OpenRouter teacher or the deterministic
   synthetic teacher.
3. Execute teacher Cypher tool calls locally through Neo4j validation.
4. Allow bounded repair attempts, then keep only interactions that pass static
   safety, execution, and task-specific checks. QA rows are compared with the
   expected answer; PDDL rows must satisfy their grounding contract and the final
   goal must be valid for the configured robot domain.
5. Render accepted traces as `ollama_native` or `custom_xml`, optionally pruning
   unnecessary failed repair turns for student training.
6. Train QLoRA student agents on grouped, leakage-checked splits.
7. Export project-local Ollama metadata and Heracles Agents example configs.

Cypher seed records remain as internal support for teacher trace generation.
The public training and evaluation surface is agent-trace based.

## Quick Start

The supported default interface is the GPU-required Docker stack. It provides
the project Python environment, Neo4j, CUDA-enabled PyTorch, and llama.cpp
without requiring host Python package installation. Ollama runs separately in
the project-local Heracles benchmark stack so inference metrics have an
isolated, stable target.

Create the local environment file and set `OPENROUTER_API_KEY` when teacher
generation is required:

```bash
cp docker/.env.example docker/.env
```

Validate the host, build the main image, and start the development services:

```bash
docker/stack.sh setup
```

Load the configured scene and inspect service status:

```bash
docker/stack.sh load-neo4j data/scenes/uhumans2_office.json
docker/stack.sh status
```

The Docker stack provides `main` and `neo4j` services. Neo4j is
available inside Docker at `neo4j://neo4j:7683` and from the host at
`neo4j://127.0.0.1:7683`; HTTP is exposed at `http://127.0.0.1:7469`. The
username is `neo4j` and the password is `neo4j_pw`. The trace generator
validates teacher Cypher tool calls against this database, so reload the
database whenever `scene_path` changes.

Run the complete demo from database loading through baseline evaluation,
student training, validation, and static HTML report generation:

```bash
docker/stack.sh demo
```

The demo uses the deterministic synthetic teacher, so it does not require an
OpenRouter key. It establishes a base-model raw-generation baseline, trains the
configured student, and runs eval-loss plus fine-tuned raw-generation validation
on the test and challenge splits. Set `DEMO_SCENE`,
`DEMO_DATA_CONFIG`, `DEMO_TEACHER_CONFIG`, or `DEMO_STUDENT_CONFIG` to use
alternative inputs.

Run `docker/stack.sh help` for the ordered first-run guide and command groups.
See [docker/README.md](docker/README.md) for the complete command reference and
[docs/ollama_export.md](docs/ollama_export.md) for model export and registration.

## Agent Records

Canonical training data is validated by `AgentTraceRecord`. Answerable examples
call `run_cypher_query`; refusal examples call no tool and answer
`<answer>WRITE_QUERY_NOT_ALLOWED</answer>`.

Each record carries `task_family`, `output_type`, and `experiment_profile`.
`cypher_qa` records use `SLDP`; `cypher_pddl` records use `PDDL` and additionally
carry a `grounding` contract. The contract declares which result aliases and
symbol types the Cypher tool must return. It prevents a syntactically correct
goal from being accepted when the query retrieved the wrong scene entities.

Teacher-generated traces are captured from real interactions. The
training-ready accepted teacher traces are written to `teacher_traces.jsonl`.
Debug copies are written to `accepted.json` and `rejected.json`; both include
the seed `real_solution`, the executed `cypher_solution`, and the parsed
`final_answer`. Student split rendering can optionally use a pruned view that
keeps the successful tool call and final answer while leaving raw traces
unchanged.

Supported render views:

- `ollama_native`
- `custom_xml`

## Workflow

Generated dataset artifacts are resolved from `scene_id` in `configs/data.yaml`
and written under `data/3dsg/{scene_id}/...`. The source 3DSG JSON is not
derived from `scene_id`; configure the user-provided Hydra/Spark DSG file with
`scene_path`.

Question sources are configured as a list, so one generated file and its later
training splits can contain multiple task families:

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
keys remain supported and are interpreted as one question set.

Extract template values from the configured 3DSG scene:

```bash
docker/stack.sh main python scripts/extract_3dsg_values.py \
  --data-config configs/data.yaml
```

Generate raw 3DSG questions from those template values:

```bash
docker/stack.sh main python scripts/generate_3dsg_seed_templates.py \
  --data-config configs/data.yaml
```

Prepare seed records from the configured raw 3DSG questions:

```bash
docker/stack.sh main python scripts/prepare_dataset.py prepare-seeds \
  --data-config configs/data.yaml
```

Check OpenRouter model availability (OpenRouter teacher only):

```bash
docker/stack.sh main python scripts/generate_traces.py preflight \
  --teacher-config configs/teacher_gpt_oss_120b.yaml
```

Generate and validate teacher traces:

```bash
docker/stack.sh main python scripts/generate_traces.py generate \
  --data-config configs/data.yaml \
  --teacher-config configs/teacher_gpt_oss_120b.yaml
```

The OpenRouter teacher receives the profile-rendered agent prompt and the
`run_cypher_query` tool schema. It does not receive seed gold Cypher or expected
answers. `max_tool_calls` defaults to `3` in the teacher config and can be overridden
with `--max-tool-calls`.

To generate traces without an LLM teacher:

```bash
docker/stack.sh main python scripts/generate_traces.py generate \
  --data-config configs/data.yaml \
  --teacher-config configs/teacher_synthetic.yaml
```

The `synthetic` provider renders the same experiment profile, inserts the seed's gold
Cypher as the tool call, executes it against Neo4j, stores the real returned rows as
the tool output, and uses the gold solution as the final answer.

Create grouped splits:

```bash
docker/stack.sh main python scripts/prepare_dataset.py split \
  --data-config configs/data.yaml \
  --render-view ollama_native
```

Splitting is stratified by task family and query type. Task types that share
normalized-equivalent Cypher queries are assigned jointly, preserving the
configured per-type proportions while keeping each equivalent query in one
split. Split and workflow reports include counts by task family.

To optimize the student dataset by pruning failed repair attempts during
split/render output and adding rejected `invalid_answer_format` traces that can
be repaired by replacing only the final answer text:

```bash
docker/stack.sh main python scripts/prepare_dataset.py split \
  --data-config configs/data.yaml \
  --render-view ollama_native \
  --optimize-student-traces
```

Evaluate traces:

```bash
docker/stack.sh main python scripts/evaluate_traces.py \
  --data-config configs/data.yaml \
  --split train
```

Train the primary student:

```bash
docker/stack.sh main python scripts/train_agent_lora.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml
```

Student training defaults to `training.loss_mode: assistant-only`, which masks
user/system/tool-response tokens from the loss and supervises only assistant
turns, including generated tool calls and final answers. The split's declared
`run_cypher_query` tool is resolved to its native JSON schema and passed once to
the tokenizer chat template; it is not copied into the stored trace messages.
To inspect the exact
rendered text and supervised token spans for a split record:

```bash
docker/stack.sh main python scripts/inspect_training_dataset.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --split train \
  --index 0
```

Evaluate the primary student:

```bash
docker/stack.sh main python scripts/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --split all \
  --mode eval-loss
```

```bash
docker/stack.sh main python scripts/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --split test \
  --mode raw-generation \
  --max-new-tokens 1024
```

Evaluate the same split and generation settings with the base student model,
without loading the trained LoRA adapter:

```bash
docker/stack.sh main python scripts/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml \
  --split test \
  --mode raw-generation \
  --baseline \
  --max-new-tokens 1024
```

Generate a static HTML workflow report for the configured scene:

```bash
docker/stack.sh main python scripts/generate_workflow_report.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml
```

See [docs/ollama_export.md](docs/ollama_export.md) for LoRA merge, GGUF
conversion, quantization, and Ollama model creation.

## Host-Only Setup

Host execution is optional and intended for environments where the Docker
stack cannot be used:

```bash
python3 -m venv .venv
source .venv/bin/activate
git submodule update --init --recursive
pip install -e '.[dev,train]'
```

Install a PyTorch build matching the host CUDA or JetPack environment
separately. Host-side GGUF conversion also requires a separate llama.cpp setup;
see the host-only section in [docs/ollama_export.md](docs/ollama_export.md).

## Heracles Benchmark

This repository does not modify or mount
`/home/coder/workspace/heracles_devel`. The equivalent isolated Docker setup,
launcher, model manifest, and experiment YAML files live under
`examples/herables_benchmark/`.

Run the current Cypher-only base-versus-fine-tuned Ollama sweep with:

```bash
examples/herables_benchmark/run_benchmark.sh --benchmarks cypher
```

The launcher also accepts `--benchmarks pddl` or `--benchmarks both` for future
PDDL-trained models. It registers enabled exports inside its own GPU Ollama
container, records Heracles local/GPU metrics and raw samples, and writes
`outputs/heracles_benchmark/model_sweep/report.html`. See the
[benchmark guide](examples/herables_benchmark/README.md).

Expected fine-tuned Ollama model names include:

- `qwen3.5-4b-finetune`
- `granite4.1-3b-finetune`

Their base-model counterparts use the `-baseline` suffix (e.g. `qwen3.5-4b-baseline`).

## Validation Gates

Use `pytest`, `ruff`, the import smoke test, and the repository naming tests
before treating a refactor as complete.

The optional host `dev` installation provides `pytest` and `ruff` for these
repository checks.

## Additional Tools

### Label room regions

Interactively add or update room classifications in a Spark DSG scene:

```bash
docker/stack.sh main python scripts/edit_3dsg_room_labels.py data/scenes/uhumans2_office.json \
  --output data/scenes/uhumans2_office.updated.json \
  --figure-output data/scenes/uhumans2_office_rooms.png
```

The editor opens a top-down view with red room bounding boxes, room symbols,
and blue object triangles. It then walks through the rooms in symbol order,
prints contained-object counts by class, and accepts either a listed
single-word label or its number. The source is preserved by default and the
result is written beside it as `*.room_labels.json`. Use `--output` to choose a
different path, `--figure-output` to save the visualization, `--no-show` for a
headless terminal, and `--list-labels` to inspect the controlled vocabulary.
