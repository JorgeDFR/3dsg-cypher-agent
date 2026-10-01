# HTML Workflow Report

The workflow report combines the artifacts and metrics from one selected
dataset and model into a static HTML file. It is intended for inspecting a run,
comparing model behavior, and tracing results back to their inputs.

Generating the report does not run missing pipeline stages. It reads the
artifacts that already exist and shows empty sections when relevant data has not
yet been produced.

## Generate a report

Use the latest dataset and latest completed model selected by the default
configuration:

```bash
docker/stack.sh main python scripts/reporting/generate_workflow_report.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml
```

The command prints the generated path, file size, and a `file://` hint. When
using Docker, the report is written through the repository bind mount. Open the
corresponding path in the host checkout.

The validation and deployment workflows regenerate the report after producing results:

```bash
docker/stack.sh demo-validate
docker/stack.sh demo-deploy
```

## Select inputs and output

Use version selectors to reproduce a particular report:

```bash
docker/stack.sh main python scripts/reporting/generate_workflow_report.py \
  --data-config configs/data.yaml \
  --dataset-version 2026-09-08-T-12-05-47-Z \
  --student-config configs/student_qwen3.5_4b.yaml \
  --model-version 2026-09-08-T-13-00-00-Z
```

| Option | Purpose |
| --- | --- |
| `--data-config` | Select the scenes, dataset family, and artifact roots |
| `--dataset-version` | Select an exact dataset version instead of the latest |
| `--student-config` | Include the selected model's training and evaluation data |
| `--model-version` | Select an exact completed model version |
| `--allow-dataset-mismatch` | Permit an intentional mismatch between model lineage and selected data |
| `--output PATH` | Override the default `workflow_report.html` destination |

Without `--output`, a model-aware report is written to the selected model's
report directory:

```text
data/datasets/{dataset_id}/{dataset_version}/reports/
  {model_family}/{model_version}/workflow_report.html
```

See [Artifact versioning](artifact_versioning.md) for selection behavior and
the complete report hierarchy.

## Report contents

The report uses tabs to separate each workflow stage:

| Tab | Contents |
| --- | --- |
| 3D Scene Graph | Per-scene summary, node counts, classes, and extracted template values |
| Generated Questions | Total, empty, and repair counts (with task families, task types, tags, questions, and seed details) |
| Traces | Accepted and rejected teacher traces, outcome counts, and rejection reasons |
| Splits | Train/eval/test/challenge distributions and per-question details |
| Train | Student configuration, eval-loss reports, and trainer-state history |
| Validation | Fine-tuned raw-generation summaries, comparisons, and example details |
| Baseline | Base-model raw-generation results and fine-tuning gaps |
| Deploy / Quantization | Optional BF16-versus-quantized metrics and examples |
| Artifacts | Paths, existence, size, and modification time for collected files |

The **Deploy / Quantization** tab appears only when matching Ollama challenge
reports exist. Generate them with `docker/stack.sh demo-deploy` or the manual
[Ollama validation workflow](validation.md#raw-generation-with-ollama).

## Interactive features

Although the output is a single static file, embedded JavaScript supports:

- tabbed navigation.
- a scene selector in the 3D Scene Graph tab.
- sortable and filterable tables.
- expandable question-type summaries.
- selectable records with detailed prompts, tool calls, and validation data.
- a quantization selector for comparing BF16 with available GGUF variants.

The report contains its styling and interaction code inline. It does not need a
running project service after generation.

The Artifacts tab lists each file once, identifies reports by filename (and path
when filenames repeat), and displays paths as plain text.

## When to regenerate

Regenerate the report after any stage that changes the selected artifacts:

- question or trace generation.
- dataset splitting or student-trace optimization.
- training or eval-loss evaluation.
- baseline or fine-tuned raw-generation validation.
- GGUF export and Ollama quantization validation.

The report is a snapshot. It does not update automatically when its source JSON,
JSONL, YAML, or GGUF files change.

## Lineage safety

When a student model is included, the generator verifies that its recorded
dataset and split hashes match the selected data. Interactive runs warn before
continuing. Non-interactive runs require `--allow-dataset-mismatch`.

Use that flag only for an intentional cross-version comparison, and keep the
selected dataset and model versions with the resulting report.
