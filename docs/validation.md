# Validation

Check stored traces, measure held-out loss, or evaluate complete tool
interactions. Commands use the [Docker stack](../docker/README.md).

## Choose a validation mode

| Target | Command or mode | Purpose |
| --- | --- | --- |
| Teacher traces | `evaluate_traces.py` | Check stored calls and answers |
| Fine-tuned model | `eval-loss` / Transformers | Measure held-out loss and perplexity |
| Base or fine-tuned model | `raw-generation` / Transformers | Test complete tool interactions |
| Exported GGUF | `raw-generation` / Ollama | Check deployed behavior |
| External model sweep | [Heracles Benchmark](../examples/herables_benchmark/README.md) | Use benchmark-owned scenes, questions, and metrics |

Query results, answer formats, and PDDL goals follow
[Validation contracts](validation_contracts.md). For offline regrading, see
[Dataset audits](dataset_audits.md).

## Automated validation and reuse

Run `docker/stack.sh demo-validate` after training. It evaluates the student and
baseline and generates the HTML report. See
[Docker automated workflows](../docker/README.md#automated-workflows).

Completed student reports are reused when data, split contents, model, runtime,
and evaluation settings match. Baseline reuse is independent of the trained
model version. Ollama reuse also checks the registered model digest.

Changed inputs or incomplete reports trigger evaluation again. Use `--force`
to repeat a completed run, including when a remote model changes under the
same identifier or revision.

## Trace validation

```bash
docker/stack.sh main python scripts/agents/evaluate_traces.py \
  --data-config configs/data.yaml --split train
```

Checks cover query safety, execution, answer matching, and PDDL grounding where
applicable.

For prepared seeds, `scripts/cypher/validate_cypher_heracles.py` provides
read-only standalone validation. Both `--input` and `--output` are required.

## Student validation

### Eval-loss

```bash
docker/stack.sh main python scripts/agents/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --split all --mode eval-loss
```

Eval-loss uses Transformers. Labels default to `training.loss_mode`.
Override with `--loss-mode full` or `--loss-mode assistant-only`.

### Raw generation with Transformers

```bash
docker/stack.sh main python scripts/agents/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --split test --mode raw-generation --max-new-tokens 1024
```

Add `--baseline` to evaluate the configured base model. Keep the split and
generation settings identical when comparing it with the fine-tuned model.

Raw generation runs bounded tool interactions and loads each required scene
into Neo4j. This replaces the active database scene.

For controlled-repair examples, evaluation provides the stored faulty call and
syntax-error feedback before asking the model for a correction. Reports mark
these cases with `controlled_repair` and `repair_template`.

### Raw generation with Ollama

[Export and register](ollama_export.md#register-models-in-the-project-stack)
the GGUF, then evaluate it:

```bash
docker/stack.sh ollama-up

docker/stack.sh main python scripts/agents/evaluate_student.py \
  --data-config configs/data.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --split challenge --mode raw-generation --runtime ollama \
  --ollama-model qwen3.5-4b-finetune-q4_k_m --max-new-tokens 1024
```

| Option | Default or purpose |
| --- | --- |
| `--ollama-temperature` | `0` |
| `--ollama-seed` | `42` |
| `--ollama-context-length` | `16384` |
| `--ollama-timeout` | Request timeout |
| `--ollama-keep-alive` | Model retention in Ollama |

Reports include the runtime and model name, allowing multiple GGUF variants
to be compared. See [Reasoning experiments](reasoning.md#compare-inference-modes)
for thinking controls.

## Compare quantized exports

Register each GGUF under a distinct name. Repeat the same Ollama evaluation
with only `--ollama-model` changed.

`docker/stack.sh demo-deploy` exports, registers, and validates the base GGUF
and quantizations configured in the student YAML. See
[GGUF registration](ollama_export.md#register-models-in-the-project-stack).

## Select versions and inspect results

Use `--dataset-version` and `--model-version` for reproducible evaluation.
[Artifact versioning](artifact_versioning.md#recorded-lineage) explains mismatch
warnings and `--allow-dataset-mismatch`.

Generate the [HTML workflow report](workflow_report.md) to compare student,
baseline, and deployment results with per-example details.
