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
  question_templates_3dsg.yaml Shared question/Cypher template families
  templates_uhumans2.yaml 3DSG-specific values generated from data/scenes/3dsg_uhumans2.json
data/raw/questions.yaml  Seed examples with gold Cypher
data/raw/uhumans2_generated_questions.yaml Generated dataset from the local uHumans2 3DSG
data/scenes/             Input 3DSG JSON files
external/heracles/       Heracles git submodule
scripts/
  extract_scene_values.py
  generate_templates.py
  heracles_neo4j.sh
  load_scene_heracles.py
  load_scene_neo4j.py
  validate_with_heracles_docker.py
  prepare_dataset.py
  validate_cypher.py
  train_lora.py
  generate_predictions.py
  summarize_validation.py
src/text2cypher_3dsg/
  dataset.py             YAML -> chat JSONL preparation and splits
  neo4j_validation.py    Static and live Neo4j validation
  train.py               LoRA/QLoRA training entry point
  infer.py               Batch prediction generation
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
git submodule update --init --recursive
pip install -e '.[dev]'
```

For fine-tuning:

```bash
pip install -e '.[train]'
```

Install the correct PyTorch build for your CUDA/ROCm environment separately if needed.

The extraction scripts use `spark-dsg` directly. Heracles is vendored as a git
submodule under `external/heracles`; the Heracles loader scripts add the submodule's
`heracles/src` directory to `PYTHONPATH` at runtime.

## Dataset Format

Each question must include `gold_cypher`. The final `solution` is kept as metadata for
your paper/evaluation, but the fine-tuning target is only Cypher.

```yaml
questions:
  - uid: 48B
    query_type: spatial_hops
    question: Tell me the number of objects in mesh places that are within 6 hops of the bag's mesh place, other than the bag itself.
    gold_cypher: |
      MATCH (bag:Object {class: "bag"})<-[:CONTAINS]-(start:MeshPlace)
      MATCH (start)-[:MESH_PLACE_CONNECTED*1..6]-(mp:MeshPlace)-[:CONTAINS]->(o:Object)
      WHERE o <> bag
      RETURN count(DISTINCT o) AS count
    solution: 28
    scene_id: example_3dsg
    read_only: true
    tags: [spatial]
```

Recommended fields for later analysis:

- `uid`: stable question id.
- `question`: natural language input.
- `gold_cypher`: canonical Cypher target.
- `query_type`: one of `class_listing`, `hierarchy`, `spatial_hops`, `positions`, etc.
- `solution`: expected answer or result rows, used for evaluation and paper artifacts.
- `read_only`: whether the query should be treated as read-only.
- `expected_result_hash`: optional hash of the executed gold-query rows.
- `tags`: optional orthogonal tags like `spatial`, `synonym`, `ordering`.
- `scene_id`: stable 3DSG id, used to audit or split by 3DSG once multiple 3DSGs are available.

Solutions use a small answer notation:

- primitive string: token, for example `O12` or `WRITE_QUERY_NOT_ALLOWED`
- primitive number: numeric value, for example `28`
- list: `[element1, element2]`
- set: `<element1, element2>`
- dict: `{key1: value1, key2: value2}`
- point: `POINT(x y z)`

Generated template solutions support two modes. `simplified` is the default and uses the
simplest useful answer:

- `object_class_centers`: point or set of points
- `object_class_inventory`: object class or set of object classes
- `object_class_count`: primitive number
- `top_object_classes_by_count`: ordered list of object classes
- `objects_in_room_symbol`, `objects_in_room_class`, `objects_in_place_symbol`: object
  symbol or set of object symbols
- `object_place_lookup`: place symbol or set of place symbols
- `nearest_object_to_mesh_place`: nearest object symbol
- `object_radius_from_point`: object symbol or set of object symbols
- `nearest_object_to_object`: nearest object symbol
- `objects_in_bounding_box`: object symbol or set of object symbols
- `mesh_place_hops`: mesh-place symbol or set of mesh-place symbols
- `room_adjacency_count`: primitive number
- `place_hops`: place symbol or set of place symbols
- `write_refusal`: `WRITE_QUERY_NOT_ALLOWED`

`informative` keeps richer row-style answers where useful, such as dictionaries with
object symbols, classes, centers, and distances. Multi-row informative answers are
rendered as sets of dictionaries, for example `<{object: O1, class: chair}>`.

## Generate Template Data

The current default dataset is generated only from `data/scenes/3dsg_uhumans2.json`. The extractor
loads the graph with `spark_dsg.DynamicSceneGraph.load`, reads nodes, positions,
semantic labels, parents/siblings, and edges through the `spark_dsg` API, and writes
scene-specific values into `configs/templates_uhumans2.yaml`.

```bash
.venv/bin/python scripts/extract_scene_values.py \
  --scene data/scenes/3dsg_uhumans2.json \
  --question-templates configs/question_templates_3dsg.yaml \
  --output configs/templates_uhumans2.yaml \
  --scene-id uhumans2
```

The extractor writes ordinary value lists plus curated case lists for restrictive
templates. The default case lists target mostly non-empty examples while preserving some
empty-answer cases:

```bash
.venv/bin/python scripts/extract_scene_values.py \
  --scene data/scenes/3dsg_uhumans2.json \
  --output configs/templates_uhumans2.yaml \
  --non-empty-ratio 0.7 \
  --max-place-object-cases 20 \
  --max-radius-cases 40 \
  --max-nearest-mesh-cases 60 \
  --max-nearest-object-cases 40 \
  --max-mesh-place-hop-cases 25 \
  --max-place-hop-cases 25
```

This avoids large Cartesian products such as every object class against every mesh place
or every radius point. The generated uHumans2 dataset is intentionally compact: it has
742 raw questions with the current defaults.

Then generate compatible raw YAML records:

```bash
.venv/bin/python scripts/generate_templates.py \
  --templates configs/templates_uhumans2.yaml \
  --output data/raw/uhumans2_generated_questions.yaml
```

The solution mode defaults to `solution_mode` in the per-3DSG config, or `simplified` if
the config omits it. Override it at generation time with:

```bash
.venv/bin/python scripts/generate_templates.py \
  --templates configs/templates_uhumans2.yaml \
  --output data/raw/uhumans2_generated_questions.informative.yaml \
  --solution-mode informative
```

For a quick smoke test:

```bash
.venv/bin/python scripts/generate_templates.py \
  --templates configs/templates_uhumans2.yaml \
  --output /tmp/uhumans2_generated_questions.yaml \
  --limit-per-template 5
```

`configs/data.yaml` currently points directly at:

```yaml
raw_questions_path: data/raw/uhumans2_generated_questions.yaml
```

Template records should still be validated before training. The shared template file is
only a bootstrap mechanism; the useful dataset should be based on real labels, symbols,
points, and bounding boxes extracted from loaded Heracles 3D scene graphs.

The current `data/scenes/3dsg_uhumans2.json` scene contains `Object`, `Place`, and `MeshPlace`
nodes, but no `Room` or `Building` nodes. Room/building templates therefore generate no
rows for this scene.

## Prepare Data

```bash
.venv/bin/python scripts/prepare_dataset.py prepare --config configs/data.yaml
.venv/bin/python scripts/prepare_dataset.py split --config configs/data.yaml
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
- `query_type`, `scene_id`, `solution`, and tags for error analysis

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

## Load uHumans2 Into Neo4j

Preferred path, using Heracles directly from the git submodule:

```bash
git submodule update --init --recursive

export NEO4J_URI=neo4j://127.0.0.1:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=neo4j_pw

python scripts/load_scene_heracles.py --scene data/scenes/3dsg_uhumans2.json
```

The Heracles wrapper uses labelspaces embedded in the Spark-DSG file when available.
This avoids mismatches with Heracles' bundled labelspace resources, which may not contain
all semantic labels used by a given 3DSG. `--object-labelspace` and `--room-labelspace`
remain available as fallbacks for 3DSGs without embedded labelspace metadata.

Heracles also ships a Docker Compose Neo4j service. This repo includes a small wrapper:

```bash
scripts/heracles_neo4j.sh up
scripts/heracles_neo4j.sh ps
scripts/heracles_neo4j.sh down
```

The Heracles Compose defaults are:

```text
NEO4J_URI=neo4j://127.0.0.1:7683
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j_pw
HTTP=http://127.0.0.1:7469
```

End-to-end validation with Docker, Heracles loading, and Cypher execution:

```bash
python scripts/validate_with_heracles_docker.py \
  --scene data/scenes/3dsg_uhumans2.json \
  --input data/processed/text2cypher.jsonl \
  --output data/processed/text2cypher.heracles.report.json
```

This command writes a JSON report with per-query execution status, expected solution,
actual solution, and solution-match status. It also prints a Rich summary table at the
end. Reproduce the same summary later with:

```bash
python scripts/summarize_validation.py --input data/processed/text2cypher.heracles.report.json
```

If Neo4j is already running, skip Docker startup:

```bash
python scripts/validate_with_heracles_docker.py \
  --skip-docker \
  --uri neo4j://127.0.0.1:7683 \
  --username neo4j \
  --password neo4j_pw
```

Fallback path, using this repo's Heracles-compatible loader:

```bash
export NEO4J_URI=neo4j://127.0.0.1:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=neo4j_pw
export NEO4J_DATABASE=neo4j

python scripts/load_scene_neo4j.py --scene data/scenes/3dsg_uhumans2.json
```

The fallback loader clears the target database by default, creates indexes, inserts:

- `Object(nodeSymbol, class, name, center, bbox_center, bbox_dim)`
- `Place(nodeSymbol, center)`
- `MeshPlace(nodeSymbol, class, center)`

and creates:

- `(Place)-[:CONTAINS]->(Object)`
- `(Place)-[:PLACE_CONNECTED]->(Place)`
- `(MeshPlace)-[:MESH_PLACE_CONNECTED]->(MeshPlace)`

After loading, run live validation:

```bash
python scripts/validate_cypher.py \
  --input data/processed/text2cypher.jsonl \
  --output data/processed/text2cypher.executed.jsonl \
  --execute
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
- breakdown by `query_type` and `tags`

This lets you compare prompt-only Gemma 4 against fine-tuned Gemma 4, and later compare
other open models with minimal repository changes.

## Near-Term TODOs

See [TODO.md](TODO.md) for the current implementation and experiment checklist.
