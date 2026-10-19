# Dataset Audits

Check question contracts or regrade saved results without new model requests.
Commands use the [Docker stack](../docker/README.md).

## Choose an operation

| Operation | Input | Result |
| --- | --- | --- |
| Contract audit | Configured question catalogs | Configuration and query checks |
| Dataset revalidation | Completed teacher dataset | Proposed or applied grade changes |
| Report rescoring | Saved results and source records | New comparison report |

These commands use saved evidence without executing Cypher. Missing or
truncated rows require fresh execution. Missing generations require fresh
inference. Neither can be treated as an empty answer.

For synthetic explanation coverage and sentence caps, use the
[reasoning audit](reasoning.md#audit-saved-traces).

## Audit validation contracts

```bash
docker/stack.sh main python scripts/cypher/audit_validation.py audit
```

This checks catalog coverage, query parsing, answer formats, PDDL goals, and
binding declarations. Use `--templates PATH ...` for alternate catalogs.
Validation files follow the discovery rules in
[Validation contracts](validation_contracts.md#configuration-files).

## Revalidate a teacher dataset

Preview grade changes for a completed dataset:

```bash
docker/stack.sh main python scripts/agents/revalidate_dataset.py \
  --data-config configs/data.yaml --dataset-version DATASET_VERSION
```

Revalidation requires current-format contracts, saved rows, sampled bindings,
and template identity. Contract refresh verifies the saved query against its
template. Regenerate incompatible datasets or records with missing provenance.

Add `--apply` after reviewing the preview:

```bash
docker/stack.sh main python scripts/agents/revalidate_dataset.py \
  --data-config configs/data.yaml --dataset-version DATASET_VERSION --apply
```

Existing files are backed up under `revalidation_backups/`. The command
refreshes questions, seeds, accepted/rejected traces, lengths, coverage, and
related generation records. Original answers and request/billing records are
preserved. Enabled teacher optimizations run against the revalidated evidence.

Changes are recorded in `reports/revalidation.report.json`. Splits are not
rebuilt automatically. Recreate them, then inspect a dataset-only report:

```bash
docker/stack.sh main python scripts/agents/prepare_dataset.py split \
  --data-config configs/data.yaml --dataset-version DATASET_VERSION

docker/stack.sh main python scripts/reporting/generate_workflow_report.py \
  --data-config configs/data.yaml --dataset-version DATASET_VERSION --dataset-only
```

Rebuilt splits can differ from those used by an existing model. Review
[recorded lineage](artifact_versioning.md#recorded-lineage) before using that
model with the updated dataset.

## Rescore a saved report

```bash
docker/stack.sh main python scripts/cypher/audit_validation.py rescore \
  --records /tmp/current-generation.report.json \
  --source-records /tmp/current-seeds.jsonl \
  --output /tmp/rescored.validation.json
```

Source records are associated by exact UID. The output includes before/after
grades, status counts, changed records, and reasons for unavailable checks.
Existing output files are protected.

Rescoring creates a separate report. Use dataset revalidation when finalized
teacher artifacts also need updating.
