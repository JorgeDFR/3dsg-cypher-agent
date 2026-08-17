# 3DSG Cypher Agent Distillation

This repository fine-tunes and distills small tool-using agents for querying
Neo4j-backed 3D Scene Graphs. The active target is an agent that answers by
calling `run_cypher_query(cypher_string: str)` through either Ollama-native tool
calls or a custom XML fallback interface.

- Default teacher: `openai/gpt-oss-120b` through OpenRouter
- Primary student: Qwen3 4B instruct, exported to Ollama for Jetson Orin NX 16GB

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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
git submodule update --init --recursive
pip install -e '.[dev]'
```

For LoRA/QLoRA training:

```bash
pip install -e '.[train]'
```

Install the PyTorch build that matches the target CUDA or JetPack environment
separately.

## Agent Records

Canonical training data is validated by `AgentTraceRecord`. Answerable examples
call `run_cypher_query`; refusal examples call no tool and answer
`<answer>WRITE_QUERY_NOT_ALLOWED</answer>`.

Teacher-generated traces are captured from real interactions. The raw accepted
trace preserves the full teacher conversation, including failed bounded repair
attempts. Student split rendering can optionally use a pruned view that keeps the
successful tool call and final answer while leaving raw traces unchanged.

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
.venv/bin/python scripts/extract_3dsg_values.py \
  --data-config configs/data.yaml
```

Generate raw 3DSG questions from those template values:

```bash
.venv/bin/python scripts/generate_3dsg_seed_templates.py \
  --data-config configs/data.yaml
```

Prepare seed records from the configured raw 3DSG questions:

```bash
.venv/bin/python scripts/prepare_dataset.py prepare-seeds \
  --data-config configs/data.yaml
```

Start Neo4j and load the same scene configured by `scene_path`:

```bash
scripts/docker/heracles_neo4j.sh up
.venv/bin/python scripts/load_3dsg_neo4j.py \
  --scene data/scenes/3dsg_uhumans2.json
```

The Neo4j helper starts a local container using Bolt on
`neo4j://127.0.0.1:7683`, HTTP on `http://127.0.0.1:7469`, username `neo4j`,
and password `neo4j_pw`. The trace generator validates teacher Cypher tool calls
against this database, so reload the database whenever `scene_path` changes.

Check OpenRouter model availability:

```bash
.venv/bin/python scripts/generate_traces.py preflight \
  --teacher-config configs/teacher_gpt_oss_120b.yaml
```

Generate and validate teacher traces:

```bash
.venv/bin/python scripts/generate_traces.py generate \
  --data-config configs/data.yaml \
  --teacher-config configs/teacher_gpt_oss_120b.yaml
```

The teacher receives the normal agent prompt, the schema context, and the
`run_cypher_query` tool schema. It does not receive seed gold Cypher or expected
answers. `max_tool_calls` defaults to `3` in the teacher config and can be
overridden with `--max-tool-calls`.

Create grouped splits:

```bash
.venv/bin/python scripts/prepare_dataset.py split \
  --data-config configs/data.yaml \
  --render-view ollama_native
```

To optimize the student dataset by pruning failed repair attempts during
split/render output:

```bash
.venv/bin/python scripts/prepare_dataset.py split \
  --data-config configs/data.yaml \
  --render-view ollama_native \
  --optimize-student-traces
```

Train the primary student:

```bash
.venv/bin/python scripts/train_agent_lora.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3_4b.yaml
```

Evaluate traces:

```bash
.venv/bin/python scripts/evaluate_traces.py \
  --data-config configs/data.yaml
```

Generate Ollama export metadata:

```bash
.venv/bin/python scripts/export_agent_ollama.py \
  --data-config configs/data.yaml
```

The export helper writes a project-local `Modelfile` scaffold. Adapter merging,
GGUF conversion, and quantization still depend on the selected base model and
local llama.cpp/Ollama toolchain.

## Heracles Agents Examples

This repository does not modify
`/home/coder/workspace/heracles_devel/heracles_agents`. Example YAML files live
under `examples/heracles_agents/` and point at the expected Ollama model names:

- `3dsg-cypher-agent-qwen3-4b-distilled`
- `3dsg-cypher-agent-granite-3b-distilled`
- `3dsg-cypher-agent-qwen3-1.7b-distilled`

## Validation Gates

Use `pytest`, `ruff`, the import smoke test, and the repository naming tests
before treating a refactor as complete.
