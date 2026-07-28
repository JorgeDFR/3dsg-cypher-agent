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
2. Ask an OpenRouter teacher for complete agent traces, not direct query text.
3. Execute candidate Cypher locally through Neo4j validation.
4. Keep only traces that pass static safety, execution, and expected-answer checks.
5. Render accepted traces as `ollama_native` or `custom_xml`.
6. Train QLoRA student agents on grouped, leakage-checked splits.
7. Export project-local Ollama metadata and Heracles Agents example configs.

Cypher seed records remain as internal support for teacher trace generation. The
public training and evaluation surface is agent-trace based.

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

Supported render views:

- `ollama_native`
- `custom_xml`

## Workflow

Prepare seed records from configured raw 3DSG questions:

```bash
.venv/bin/python scripts/prepare_dataset.py prepare-seeds \
  --config configs/data.yaml
```

Check OpenRouter model availability:

```bash
.venv/bin/python scripts/generate_traces.py preflight \
  --teacher-config configs/teacher_openrouter_gpt_oss_120b.yaml
```

Generate and validate teacher traces:

```bash
.venv/bin/python scripts/generatetraces.py generate \
  --teacher-config configs/teacher_openrouter_gpt_oss_120b.yaml \
  --data-config configs/data.yaml \
  --input data/3dsg/seeds/cypher_seed_records.jsonl \
  --output data/3dsg/agent/traces_openrouter.accepted.jsonl \
  --rejected-output data/3dsg/agent/traces_openrouter.rejected.json
```

Create grouped splits:

```bash
.venv/bin/python scripts/prepare_dataset.py split \
  --config configs/data.yaml \
  --render-view ollama_native
```

Train the primary student:

```bash
.venv/bin/python scripts/train_agent_lora.py \
  --model-config configs/student_qwen3_4b.yaml \
  --train data/3dsg/agent/splits/train.ollama_native.jsonl \
  --eval data/3dsg/agent/splits/eval.ollama_native.jsonl
```

Evaluate traces:

```bash
.venv/bin/python scripts/evaluate_traces.py \
  --input data/3dsg/agent/splits/test.ollama_native.jsonl \
  --output data/3dsg/agent/reports/test.agent.report.json
```

Generate Ollama export metadata:

```bash
.venv/bin/python scripts/export_agent_ollama.py \
  --config configs/data.yaml
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
