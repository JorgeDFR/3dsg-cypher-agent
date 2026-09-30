# Artifact Versioning

Generated datasets and fine-tuned models use explicit lineage so
a result can be traced back to its scene and split content. Versions are
readable UTC timestamps such as `2026-09-08-T-12-05-47-Z`.

## Version hierarchy

```text
data/datasets/{dataset_id}/{dataset_version}/
|-- scenes/{scene_id}/scene_template_values.yaml
|-- ... dataset-level questions, seeds, and splits
|-- traces/
|   |-- accepted.json
|   |-- rejected.json
|   `-- teacher_traces.jsonl
`-- reports/

outputs/lora/{model_family}/{model_version}/
|-- ... fine-tuned weights and tokenizer artifacts
`-- metadata.yaml
```

The exact generated filenames inside a dataset version depend on the configured
question sets and completed stages. Commands resolve them from
`configs/data.yaml`. Avoid hard-coding paths in automation when a version
selector is available.

## Selection rules

| Artifact | Default behavior | Explicit selector |
| --- | --- | --- |
| Dataset | Extraction creates a new version. Later commands use latest | `--dataset-version` |
| Teacher traces | One set per dataset. Generation forks the dataset if traces or splits already exist | `--dataset-version` selects prepared inputs. `--teacher` labels the teacher |
| LoRA model | Training creates a new version. Evaluation/export use latest completed | `--model-version` |

The source 3DSG files are not inferred from their identifiers. Each item in the
data configuration's `scenes` list explicitly pairs `scene_id` with
`scene_path`. `dataset_id` identifies their combined artifact family.

Teacher generation never replaces an existing trace set. If `traces/` or
`splits/` already contains artifacts (including a partial teacher run), it creates
a new dataset version, copies prepared questions, seed records, scene values,
and other input files, and generates fresh traces there. It does not copy
splits or validation reports. The summary prints the new version. Pass that
version to `prepare_dataset.py split` when using a pinned data configuration.
The source dataset and models trained from it remain unchanged.

Historical datasets with nested teacher versions remain readable through their
recorded active teacher. New generation always uses the single-teacher layout.
Historical reports are also readable from `reports/dataset` and `reports/models`.
New reports use the flattened layout below.

## Inspect and select versions

List available dataset and LoRA identifiers, and the selected dataset's teacher:

```bash
docker/stack.sh main python scripts/list_artifact_versions.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml
```

Pass the printed identifiers to generation, splitting, training, evaluation,
reporting, or export commands as appropriate.

## Recorded lineage

A trained model records:

- dataset and model version identifiers.
- hashes of the exact training and evaluation splits.
- source configuration records.
- training metrics and selected checkpoint.
- fingerprints of key fine-tuned weights and tokenizer artifacts.

When evaluation or reporting selects different data, the command warns in an
interactive session. Non-interactive jobs must pass
`--allow-dataset-mismatch` to continue.

## Report layout

Dataset reports and model-family folders share the reports root:

```text
data/datasets/{dataset_id}/{dataset_version}/reports/
|-- split.report.json
|-- test.agent.report.json
`-- {model_family}/
    |-- baseline/
    `-- {model_version}/
```

This layout keeps trace and split reports with the dataset while routing model
metrics to the selected model family and version.

## GGUF export policy

GGUF artifacts are intentionally not timestamped because they are large:

```text
outputs/ollama/{model_name}/
```

Starting an export from a different LoRA version replaces that model's previous
export directory. `metadata.yaml` stores the selected source model and dataset
lineage so Ollama validation reports can still be routed to the corresponding
model-version report directory.

Preserve or copy an export before switching source versions if both sets of
GGUF files must remain available simultaneously.

## Migrate legacy flat artifacts

For a checkout that still uses the former flat directories, preview the
one-time migration:

```bash
docker/stack.sh main python scripts/migrate_versioned_artifacts.py
```

Apply it only after reviewing the proposed changes:

```bash
docker/stack.sh main python scripts/migrate_versioned_artifacts.py --apply
```
