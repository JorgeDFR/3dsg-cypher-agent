# 3DSG Text-to-Cypher Fine-Tuning

This repository prepares and fine-tunes LLMs for the text-to-Cypher component of an
agentic framework over Neo4j 3D Scene Graph databases.

The initial target model configuration is Gemma 4, but the scripts are model-agnostic:
to test another Hugging Face causal/chat model, add another `configs/model_*.yaml` file
and keep the dataset format unchanged.

## Scope

This repo focuses only on:

1. Natural language question -> Cypher query.
2. Validation of generated Cypher against 3DSG-specific rules and optionally Neo4j.
3. LoRA/QLoRA fine-tuning of open models.

It intentionally does not train the downstream answer/PDDL generation component.

## Repository Layout

```text
configs/
  data.yaml              Dataset paths and split settings
  model_gemma4.yaml      Default model/training config
  schema_3dsg.yaml       Heracles-style Neo4j 3DSG schema prompt
  templates_3dsg.yaml    Template families for synthetic data generation
data/raw/questions.yaml  Seed examples with gold Cypher
scripts/
  generate_templates.py
  prepare_dataset.py
  validate_cypher.py
  train_lora.py
  generate_predictions.py
  summarize_validation.py
src/sg_text2cypher/
  dataset.py             YAML -> chat JSONL preparation and splits
  neo4j_validation.py    Static and live Neo4j validation
  train.py               LoRA/QLoRA training entry point
  infer.py               Batch prediction generation
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For fine-tuning:

```bash
pip install -e '.[train]'
```

Install the correct PyTorch build for your CUDA/ROCm environment separately if needed.

## Dataset Format

Each question must include `gold_cypher`. The final `solution` is kept as metadata for
your paper/evaluation, but the fine-tuning target is only Cypher.

```yaml
questions:
  - uid: 48B
    name: Objects near bag
    question: Tell me the number of objects in mesh places that are within 6 hops of the bag's mesh place, other than the bag itself.
    gold_cypher: |
      MATCH (bag:Object {class: "bag"})<-[:CONTAINS]-(start:MeshPlace)
      MATCH (start)-[:MESH_PLACE_CONNECTED*1..6]-(mp:MeshPlace)-[:CONTAINS]->(o:Object)
      WHERE o <> bag
      RETURN count(DISTINCT o) AS count
    solution: "28"
    task_type: spatial_hops
    tags: [spatial]
```

Recommended fields for later analysis:

- `uid`: stable question id.
- `question`: natural language input.
- `gold_cypher`: canonical Cypher target.
- `query_type` / `task_type`: one of `class_listing`, `hierarchy`, `spatial_hops`,
  `positions`, etc.
- `database`: stable database/domain name, for example `heracles_scene_graph`.
- `read_only`: whether the query should be treated as read-only.
- `expected_result_hash`: optional hash of the executed gold-query rows.
- `tags`: optional orthogonal tags like `spatial`, `synonym`, `ordering`.
- `scene_id`: add this when using multiple scene graphs, so splits can be audited.

## Generate Template Data

The initial hand-written examples are intentionally small. To scale the text-to-Cypher
dataset, edit `configs/templates_3dsg.yaml` and generate compatible YAML records:

```bash
python scripts/generate_templates.py \
  --templates configs/templates_3dsg.yaml \
  --output data/raw/generated_questions.yaml
```

For a quick smoke test:

```bash
python scripts/generate_templates.py \
  --templates configs/templates_3dsg.yaml \
  --output data/raw/generated_questions.yaml \
  --limit-per-template 5
```

Then include generated records in `configs/data.yaml`:

```yaml
additional_raw_questions_paths:
  - data/raw/generated_questions.yaml
```

Template records should still be validated before training. The template file is only a
bootstrap mechanism; the useful dataset should be based on real labels, symbols, points,
and bounding boxes extracted from loaded Heracles scenes.

## Prepare Data

```bash
python scripts/prepare_dataset.py prepare --config configs/data.yaml
python scripts/prepare_dataset.py split --config configs/data.yaml
```

Outputs:

```text
data/processed/text2cypher.jsonl
data/splits/train.jsonl
data/splits/eval.jsonl
data/splits/test.jsonl
```

Each JSONL row contains:

- `question`
- `cypher`
- `schema`
- `messages`, ready for chat fine-tuning
- metadata for task/error analysis

## Validate Gold Cypher

Static validation checks for common constraints:

- no APOC
- no deprecated `distance(...)`
- no write keywords unless explicitly allowed
- labels must be in the configured 3DSG schema
- relationships must be in the configured 3DSG schema
- properties must be in the configured 3DSG schema
- variable-length paths must have an upper bound, for example `*1..5`

```bash
python scripts/validate_cypher.py \
  --input data/processed/text2cypher.jsonl \
  --output data/processed/text2cypher.validated.jsonl \
  --static-only
```

To validate against Neo4j with `EXPLAIN`, configure `.env` or pass credentials:

```bash
export NEO4J_URI=neo4j://127.0.0.1:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=neo4j_pw
export NEO4J_DATABASE=neo4j

python scripts/validate_cypher.py \
  --input data/processed/text2cypher.jsonl \
  --output data/processed/text2cypher.neo4j.jsonl
```

To execute read queries and capture returned rows:

```bash
python scripts/validate_cypher.py \
  --input data/processed/text2cypher.jsonl \
  --output data/processed/text2cypher.executed.jsonl \
  --execute
```

When `--execute` is used, validation output includes `rows` and `result_hash`. The hash
is intended for later result-equivalence evaluation between gold and predicted Cypher.

Summarize validation:

```bash
python scripts/summarize_validation.py --input data/processed/text2cypher.neo4j.jsonl
```

## Fine-Tune

Edit `configs/model_gemma4.yaml` and set `model.base_model` to the exact Gemma 4
checkpoint or local path you are using.

```bash
python scripts/train_lora.py \
  --model-config configs/model_gemma4.yaml \
  --train data/splits/train.jsonl \
  --eval data/splits/eval.jsonl
```

The default config uses QLoRA-style 4-bit loading and LoRA targets commonly used by
Gemma/Llama/Qwen-style decoder models:

```yaml
lora:
  target_modules: auto
```

For a future model, create a new file such as `configs/model_qwen.yaml` and change:

```yaml
model:
  base_model: Qwen/...
  tokenizer: null
  trust_remote_code: true
```

If the architecture needs different LoRA modules, override:

```yaml
lora:
  target_modules: [q_proj, k_proj, v_proj, o_proj]
```

## Generate and Validate Predictions

After training, point `model.base_model` at the adapter/output directory or a merged model,
then run:

```bash
python scripts/generate_predictions.py \
  --model-config configs/model_gemma4.yaml \
  --input data/splits/test.jsonl \
  --output data/processed/test_predictions.jsonl
```

Validate predicted Cypher:

```bash
python scripts/validate_cypher.py \
  --input data/processed/test_predictions.jsonl \
  --output data/processed/test_predictions.validated.jsonl \
  --cypher-field predicted_cypher
```

## Article-Oriented Metrics

For the text-to-Cypher component, report at least:

- syntax-valid rate
- schema-valid/executable rate
- timeout rate
- forbidden-function rate
- result-returning rate
- exact/result-hash match, once expected tool results are available
- breakdown by `task_type` and `tags`

This lets you compare prompt-only Gemma 4 against fine-tuned Gemma 4, and later compare
other open models with minimal repository changes.

## Near-Term TODOs

See [TODO.md](TODO.md) for the current implementation and experiment checklist.
