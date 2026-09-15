# Artifact Versioning

Generated datasets, teacher runs, and trained adapters use explicit lineage so
a result can be traced back to its scene and split content. Versions are
readable UTC timestamps such as `2026-09-08-T-12-05-47-Z`.

## Version hierarchy

```text
data/datasets/{dataset_id}/{dataset_version}/
|-- scenes/{scene_id}/scene_template_values.yaml
|-- ... dataset-level questions, seeds, and splits
|-- traces/{teacher_name}/{teacher_run}/
|   |-- accepted.json
|   |-- rejected.json
|   `-- teacher_traces.jsonl
`-- reports/

outputs/lora/{model_family}/{model_version}/
|-- ... adapter and tokenizer artifacts
`-- metadata.yaml
```

The exact generated filenames inside a dataset version depend on the configured
question sets and completed stages. Commands resolve them from
`configs/data.yaml`; avoid hard-coding paths in automation when a version
selector is available.

## Selection rules

| Artifact | Default behavior | Explicit selector |
| --- | --- | --- |
| Dataset | Extraction creates a new version; later commands use latest | `--dataset-version` |
| Teacher trace run | Generation creates a new run; splitting uses the active run | `--teacher`, `--teacher-run` |
| LoRA model | Training creates a new version; evaluation/export use latest completed | `--model-version` |

The source 3DSG files are not inferred from their identifiers. Each item in the
data configuration's `scenes` list explicitly pairs `scene_id` with
`scene_path`; `dataset_id` identifies their combined artifact family.

## Inspect and select versions

List available dataset, teacher, and LoRA identifiers and show which versions
are currently selected:

```bash
docker/stack.sh main python scripts/list_artifact_versions.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml
```

Pass the printed identifiers to generation, splitting, training, evaluation,
reporting, or export commands as appropriate.

## Recorded lineage

A trained model records:

- dataset and model version identifiers;
- hashes of the exact training and evaluation splits;
- source configuration records;
- training metrics and selected checkpoint;
- fingerprints of key adapter and tokenizer artifacts.

When evaluation or reporting selects different data, the command warns in an
interactive session. Non-interactive jobs must pass
`--allow-dataset-mismatch` to continue.

## Report layout

Dataset and model reports remain isolated:

```text
data/datasets/{dataset_id}/{dataset_version}/reports/
|-- dataset/
`-- models/{model_family}/
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
