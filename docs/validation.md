# Validation

Validation operates at three levels: dataset trace checks, student eval-loss,
and end-to-end raw generation. Raw generation can use either Transformers or a
GGUF model registered in Ollama.

## Automated validation and reuse

Run `docker/stack.sh demo-validate` after training. It evaluates loss, validates
test/challenge generation for both the student model and baseline, and generates the
HTML report. It does not create data or train a student model.

`evaluate_student.py` automatically skips a completed report when the dataset,
split contents, model, runtime, and evaluation settings match. Baseline reuse
is independent of fine-tuned model versions and training hyperparameters. Ollama
validation also checks the registered model digest. Changed inputs, incomplete
reports, or unreadable reports trigger evaluation again. Pass `--force` to
explicitly repeat a validation. Older reports without a validation fingerprint
are evaluated once to establish that fingerprint. For remotely hosted HF
models, reuse identifies the configured model/revision. Use `--force` if its
upstream contents change under the same identifier.

## Validation matrix

| Target | Mode/runtime | Main question answered |
| --- | --- | --- |
| Teacher traces | `evaluate_traces.py` | Are stored tool calls and answers valid? |
| Student model | `eval-loss` / Transformers | How well does the model fit held-out rendered traces? |
| Base or fine-tuned model | `raw-generation` / Transformers | Can the model complete the full tool interaction? |
| Exported GGUF | `raw-generation` / Ollama | Does the deployed artifact preserve behavior? |
| External model sweep | Heracles Benchmark | How does the model perform on benchmark-owned questions and metrics? |

## Trace validation

Evaluate stored traces from one split:

```bash
docker/stack.sh main python scripts/agents/evaluate_traces.py \
  --data-config configs/data.yaml \
  --split train
```

Trace checks cover static safety, Cypher execution, output parsing, QA answer
matching, and PDDL grounding and goal validity where applicable.

## Student validation

### Eval-loss

Compute held-out loss and perplexity for the fine-tuned model:

```bash
docker/stack.sh main python scripts/agents/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --split all \
  --mode eval-loss
```

Eval-loss uses Transformers only. Its token labels default to
`training.loss_mode` in the student config. Override them with
`--loss-mode full` or `--loss-mode assistant-only`.

### Raw generation with Transformers

Evaluate the fine-tuned model end to end:

```bash
docker/stack.sh main python scripts/agents/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --split test \
  --mode raw-generation \
  --max-new-tokens 1024
```

Evaluate the configured base model with the same split and generation settings:

```bash
docker/stack.sh main python scripts/agents/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --split test \
  --mode raw-generation \
  --baseline \
  --max-new-tokens 1024
```

Raw-generation validation runs a bounded query/tool/final-answer interaction
and reuses the dataset's tool schema, Neo4j checks, retry policy, and task
validators. Records are grouped by `scene_id` before execution, and Neo4j is
cleared and loaded only when that scene changes. A complete validation pass
therefore performs one database load per represented scene.

For controlled-repair records, the standalone Cypher generation and end-to-end
checks start after the stored mutated call and Neo4j syntax-error response, so
the first generated query must be a correction. The final-answer check receives
that same repair context plus the corrected Cypher call and successful tool
response. Reports include `controlled_repair` and `repair_template` fields for
filtering these cases.

### Raw generation with Ollama

First [export and register](ollama_export.md#register-models-in-the-project-stack)
the desired GGUF. Then start the optional service and evaluate it:

```bash
docker/stack.sh ollama-up
docker/stack.sh load-neo4j data/scenes/uhumans2_office.json

docker/stack.sh main python scripts/agents/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --split challenge \
  --mode raw-generation \
  --runtime ollama \
  --ollama-model qwen3.5-4b-finetune-q4_k_m \
  --max-new-tokens 1024
```

Ollama evaluation uses non-streaming native tool calls. Defaults are
deterministic (`temperature=0`, `seed=42`) with a context length of `16384`.
Relevant overrides include:

- `--ollama-temperature`
- `--ollama-seed`
- `--ollama-timeout`
- `--ollama-context-length`
- `--ollama-keep-alive`

The report name includes the runtime and model name, so multiple GGUF variants
can be compared without overwriting one another.

## Compare BF16 and quantized exports

Register each GGUF under a distinct Ollama model name, then repeat the same
raw-generation command with only `--ollama-model` changed. For example:

| GGUF | Suggested model name |
| --- | --- |
| BF16 | `qwen3.5-4b-finetune-bf16` |
| Q8_0 | `qwen3.5-4b-finetune-q8_0` |
| Q4_K_M | `qwen3.5-4b-finetune-q4_k_m` |

`docker/stack.sh demo-deploy` automates export, registration, challenge
validation of all three variants, and report regeneration.

## Dataset lineage checks

Model metadata records the dataset version and exact split hashes used during
training. Evaluation against different data behaves as follows:

- Interactive execution displays a warning and asks for confirmation.
- Non-interactive execution stops unless `--allow-dataset-mismatch` is passed.
- `--model-version` and `--dataset-version` select exact artifacts instead of
  the latest completed versions.

Treat `--allow-dataset-mismatch` as an explicit experimental choice and record
the compared versions with the results.

## Workflow report

After completing the desired validation modes, generate the
[HTML workflow report](workflow_report.md) to inspect fine-tuned and baseline
results, per-example details, and optional quantization comparisons together.

## External benchmark

Project-split validation and Heracles Benchmark answer different questions:

- Use this guide for generated splits and the matching local Neo4j scene.
- Use the [Heracles Benchmark integration](../examples/herables_benchmark/README.md)
  for benchmark-owned scenes, questions, manifests, metrics, and reports.
