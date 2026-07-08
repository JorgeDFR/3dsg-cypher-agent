# TODO

This file tracks practical next steps for turning the repo into a reproducible
text-to-Cypher fine-tuning experiment for Heracles/Hydra 3D scene graphs.

## Done

- [x] Move 3DSG files into `data/scenes/`.
- [x] Add Heracles as `external/heracles` git submodule.
- [x] Use `spark_dsg` for 3DSG extraction instead of raw JSON parsing.
- [x] Generate the first uHumans2 dataset from `data/scenes/3dsg_uhumans2.json`.
- [x] Split shared question templates from per-3DSG values:
  - `configs/question_templates_3dsg.yaml`
  - `configs/templates_uhumans2.yaml`
- [x] Add simplified and informative solution modes, with simplified as default.
- [x] Add compact answer notation for solutions.
- [x] Add curated case lists for restrictive templates to keep mostly non-empty answers.
- [x] Reduce hop-query generation to about 50 questions per hop family.
- [x] Fix hop gold Cypher to exclude the start node from returned results.
- [x] Add object class inventory, top object class count, nearest object-to-object, and balanced spatial/radius templates.
- [x] Add Heracles-backed Neo4j loader wrapper using the submodule.
- [x] Use embedded Spark-DSG labelspaces when loading through Heracles.
- [x] Add Docker/Heracles validation script.
- [x] Add JSON validation reports with expected/actual solution comparison and Rich summaries.
- [x] Support reproducing report summaries with `scripts/summarize_validation.py`.

## Dataset

- [ ] Load 3-5 additional Heracles/Hydra 3D scene graphs and assign stable `scene_id` values.
- [ ] Export object classes, room symbols, place symbols, mesh-place symbols, useful anchors, and curated cases for each additional 3DSG.
- [ ] Split train/eval/test by `scene_id` once multiple 3DSGs are available.
- [ ] Expand room/building templates once a 3DSG with `Room` and `Building` nodes is available.
- [ ] Add more relationship-reasoning templates that combine containment, connectivity, and aggregation.
- [ ] Add more abstract multi-step templates, for example "rooms/places with the fewest objects" and "nearest class after filtering by containment".
- [ ] Add more synonym/paraphrase coverage without reintroducing large Cartesian-product duplication.
- [ ] Decide whether empty-answer examples should be capped globally per query family.

## Evaluation

- [x] Compare executed query results against expected `solution`.
- [x] Report metrics by `query_type`.
- [x] Normalize result rows into the same solution notation used by the dataset.
- [ ] Add tag-level metrics to the JSON report summary.
- [ ] Add numeric tolerance for point coordinates and distance outputs where informative solutions include distances.
- [ ] Add an option to save a compact failures-only report.
- [ ] Add a guardrail check for overly broad queries, such as missing `LIMIT` in nearest-neighbor patterns.
- [ ] Add a read-only Neo4j user for live validation and prediction evaluation.

## Fine-Tuning

- [ ] Confirm the exact Gemma checkpoint name/path and update `configs/model_gemma4.yaml`.
- [ ] Run a tiny overfit test on 20-50 examples to verify the training loop.
- [ ] Train the first LoRA adapter on the current validated dataset.
- [ ] Evaluate prompt-only Gemma vs fine-tuned Gemma on the same held-out set.
- [ ] Add at least one alternative model config, such as Qwen Coder or Llama, without changing dataset code.
- [ ] Record training settings, GPU type, runtime, seed, dataset hash, and adapter output path.

## Paper Artifacts

- [ ] Create tables for static-valid, executable, solution-match, timeout, and safety-pass rates.
- [ ] Create per-query-family plots from validation reports.
- [ ] Save representative failure cases for error analysis.
- [ ] Document examples where fine-tuning helps and where prompt-only performance is already sufficient.
- [ ] Preserve the exact raw dataset, split files, config files, and validation reports used for each reported experiment.
