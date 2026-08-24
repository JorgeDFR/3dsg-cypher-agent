# 3DSG Cypher Agent Distillation

This repository fine-tunes and distills small tool-using agents for querying
Neo4j-backed 3D Scene Graphs. The active target is an agent that answers by
calling `run_cypher_query(cypher_string: str)` through either Ollama-native tool
calls or a custom XML fallback interface.

- Default teacher: `openai/gpt-oss-120b` through OpenRouter
- Primary student: `qwen-3.5-4b` through Hugging Face, and exported to Ollama

## Scope

The active workflow is:

1. Generate or load validated 3DSG Cypher seed records.
2. Run an OpenRouter teacher as a normal tool-using agent conversation.
3. Execute teacher Cypher tool calls locally through Neo4j validation.
4. Allow bounded repair attempts, then keep only interactions that pass static
   safety, execution, and expected-answer checks.
5. Render accepted traces as `ollama_native` or `custom_xml`, optionally pruning
   unnecessary failed repair turns for student training.
6. Train QLoRA student agents on grouped, leakage-checked splits.
7. Export project-local Ollama metadata and Heracles Agents example configs.

Cypher seed records remain as internal support for teacher trace generation.
The public training and evaluation surface is agent-trace based.

## Quick Start

The supported default interface is the GPU-required Docker stack. It provides
the project Python environment, Neo4j, Ollama, CUDA-enabled PyTorch, and
llama.cpp without requiring host Python package installation.

Create the local environment file and set `OPENROUTER_API_KEY` when teacher
generation is required:

```bash
cp docker/.env.example docker/.env
```

Validate the host, build the main image, and start all three services:

```bash
docker/stack.sh setup
```

Load the configured scene and inspect service status:

```bash
docker/stack.sh load-neo4j
docker/stack.sh status
```

Run `docker/stack.sh help` for the ordered first-run guide and command groups.
See [docs/docker.md](docs/docker.md) for the complete command reference and
[docs/ollama_export.md](docs/ollama_export.md) for model export and registration.

## Agent Records

Canonical training data is validated by `AgentTraceRecord`. Answerable examples
call `run_cypher_query`; refusal examples call no tool and answer
`<answer>WRITE_QUERY_NOT_ALLOWED</answer>`.

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

Reload Neo4j whenever `scene_path` changes:

```bash
docker/stack.sh load-neo4j data/scenes/3dsg_uhumans2.json
```

The Docker stack provides `main`, `neo4j`, and `ollama` services. Neo4j is
available inside Docker at `neo4j://neo4j:7683` and from the host at
`neo4j://127.0.0.1:7683`; HTTP is exposed at `http://127.0.0.1:7469`. The
username is `neo4j` and the password is `neo4j_pw`. The trace generator
validates teacher Cypher tool calls against this database, so reload the
database whenever `scene_path` changes.

Check OpenRouter model availability:

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

The teacher receives the normal agent prompt, the schema context, and the
`run_cypher_query` tool schema. It does not receive seed gold Cypher or expected
answers. `max_tool_calls` defaults to `3` in the teacher config and can be
overridden with `--max-tool-calls`.

Create grouped splits:

```bash
docker/stack.sh main python scripts/prepare_dataset.py split \
  --data-config configs/data.yaml \
  --render-view ollama_native
```

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
turns, including generated tool calls and final answers. To inspect the exact
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
  --max-new-tokens 512
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

## Heracles Agents Examples

This repository does not modify
`/home/coder/workspace/heracles_devel/heracles_agents`. Example YAML files live
under `examples/heracles_agents/` and point at the expected Ollama model names:

- `qwen3.5-4b-3dsg-cypher-agent-lora`
- `granite-3b-3dsg-cypher-agent-distiled`

## Validation Gates

Use `pytest`, `ruff`, the import smoke test, and the repository naming tests
before treating a refactor as complete.

The optional host `dev` installation provides `pytest` and `ruff` for these
repository checks.
