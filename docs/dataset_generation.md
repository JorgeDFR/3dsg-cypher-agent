# Dataset Generation

Prepare scenes and questions, generate teacher traces, then create student
splits. Commands use the [Docker stack](../docker/README.md).

For the automated workflow, run `docker/stack.sh demo-dataset`. Use the steps
below to customize individual stages. See [Artifact versioning](artifact_versioning.md)
for output paths and explicit version selection.

## Inputs

### Dataset configuration

Edit `configs/data.yaml` to select the scenes and questions to generate.

| Key | Meaning |
| --- | --- |
| `dataset_id` | Required dataset identifier used in artifact paths |
| `scenes` | One or more `scene_id` and Hydra/Spark DSG `scene_path` pairs |
| `schema_path` | 3DSG schema used by template generation and validation |
| `question_sets` | One or more experiment profile and question-template pairs |
| `max_per_task_type` | Optional cap on generated records per task type |
| `split` | Train, evaluation, test, and challenge selection settings |

Each question set pairs `experiment_profile_path` with
`question_templates_path`. See the [default configuration](../configs/data.yaml)
for a dataset that combines QA and PDDL questions.

### Source scenes

List each source scene explicitly:

```yaml
scenes:
  - scene_id: uhumans2_office
    scene_path: data/scenes/uhumans2_office.json
  - scene_id: spot_demo
    scene_path: data/scenes/spot_demo.json
```

For a single-scene dataset, keep one entry in `scenes`.

Scenes can have different layer layouts. Templates that require a missing
layer are skipped for that scene. For example, room questions are skipped when
the graph has no `Room` layer.

### Question sampling

Question wording, queries, and sampling settings are defined in
[`configs/question_templates/`](../configs/question_templates/).

| Setting | Purpose |
| --- | --- |
| `target_questions` | Total question budget per type across compatible scenes |
| `sampling.empty_ratio` | Share of questions with empty results, including zero counts |
| `sampling.max_solution_size` | Maximum number of entities in a sampled solution |
| `sampling.max_result_rows` | Maximum projected result rows for a sampled solution |

Question budgets and empty-result ratios apply to the combined dataset.
Questions are spread across compatible scenes according to available values.
Empty-result questions are excluded from controlled Cypher repair.

Size limits skip oversized candidates during sampling. Individual templates
can override these limits. Fixed inventory and aggregate questions are exempt.

Answer formats and grounding rules are configured in
[`configs/question_templates/validation/`](../configs/question_templates/validation/).
See [Validation contracts](validation_contracts.md) when
customizing these rules.

## Prepare a dataset version

### 1. Start the services

```bash
docker/stack.sh up
```

Teacher generation loads each configured scene into Neo4j automatically.
This replaces the database's current scene. To load a scene for inspection:

```bash
docker/stack.sh main python scripts/graph/load_3dsg_neo4j.py \
  --data-config configs/data.yaml --scene-id spot_demo
```

Use [Scene utilities](scene_tools.md) to edit room labels or repair connections
before generating data.

### 2. Extract scene values

```bash
docker/stack.sh main python scripts/graph/extract_3dsg_values.py \
  --data-config configs/data.yaml
```

Extraction creates a new dataset version. Later commands select the latest
version unless `--dataset-version DATASET_VERSION` is supplied.

### 3. Generate questions

```bash
docker/stack.sh main python scripts/cypher/generate_3dsg_seed_templates.py \
  --data-config configs/data.yaml
```

Each question-template file controls its own sampling. Set
`sampling.repair_cypher_ratio` to select a share of eligible questions for
controlled repair. The default is `0.2`. Override it for one run with
`--repair-cypher-ratio`. The total question count stays the same.

Empty-result and safety-refusal questions are excluded from controlled repair.
See [Teacher generation](teacher_generation.md#controlled-cypher-repair) for
how repair examples are generated.

### 4. Prepare seeds

```bash
docker/stack.sh main python scripts/agents/prepare_dataset.py prepare-seeds \
  --data-config configs/data.yaml
```

This validates the questions and prepares reference queries and answers for
teacher validation. Scene label information is taken from each source scene.

## Generate teacher traces

Follow [Teacher generation](teacher_generation.md) to select a teacher and
collect validated interactions. To compare remote teachers before a full run,
use [Teacher checks and pilots](teacher_checks.md).

Return here with the dataset version printed by generation.

## Create student splits

```bash
docker/stack.sh main python scripts/agents/prepare_dataset.py split \
  --data-config configs/data.yaml \
  --dataset-version DATASET_VERSION \
  --render-view ollama_native
```

Use `--render-view custom_xml` for the XML interface instead of native tool calls.

Splitting uses accepted teacher traces and balances task families and query
types. Related queries are grouped to reduce leakage. Exact per-type count
requirements can require separating some variants.

Configure the following under `split` in `configs/data.yaml`:

| Setting | Purpose |
| --- | --- |
| `train_size`, `eval_size`, `test_size` | Proportions for the standard splits |
| `seed` | Reproducible split selection |
| `challenge_task_types` | Question types reserved for challenge evaluation |
| `challenge_questions_per_task_type` | Dataset-wide question budget for each challenge type |

Challenge types are excluded from train, evaluation, and test. Their question
budget overrides the template's `target_questions`, subject to
`max_per_task_type` when set.

Scene-held-out splitting is not supported. Keep `split_by_scene: false`.
All configured scenes may appear in each split.

## Inspect and use the dataset

| Task | Guide |
| --- | --- |
| Browse questions, traces, and split coverage | [HTML workflow report](workflow_report.md) |
| Check stored traces | [Trace validation](validation.md#trace-validation) |
| Regrade saved evidence | [Dataset audits](dataset_audits.md) |
| Train a student | [Student training](training.md) |
