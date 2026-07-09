# TODO

This file tracks practical next steps for turning the repo into a reproducible
text-to-Cypher fine-tuning experiment for Heracles/Hydra 3D scene graphs.

## Done

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
- [x] Update training and inference code for current Transformers/TRL APIs.
- [x] Add focused unit coverage for training config, Gemma 4 handling, evaluation, and validation defaults.

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
- [x] Accept common unaliased Neo4j return-column shapes, such as `count(o)` and `o.center`.
- [x] Mark unexpected executable result shapes as solution mismatches with `result_shape_error`.
- [ ] Add tag-level metrics to the JSON report summary.
- [ ] Add numeric tolerance for point coordinates and distance outputs where informative solutions include distances.
- [ ] Add an option to save a compact failures-only report.
- [ ] Add a guardrail check for overly broad queries, such as missing `LIMIT` in nearest-neighbor patterns.
- [ ] Add a read-only Neo4j user for live validation and prediction evaluation.

## Fine-Tuning

- [x] Set the default Gemma config to `google/gemma-4-E2B-it`.
- [x] Update `configs/model_gemma4.yaml` to use `dtype`, `max_length`, and `warmup_steps` names expected by the current stack.
- [x] Add Gemma 4 E2B/E4B QLoRA preparation that avoids fp32 upcast of the huge frozen PLE embeddings.
- [x] Resolve Gemma 4 E2B/E4B LoRA auto-targets to supported inner text `linear` modules instead of unsupported `Gemma4ClippableLinear` wrappers.
- [x] Add `--model-source base|finetuned` to prediction generation.
- [x] Load fine-tuned predictions by attaching the PEFT adapter from `training.output_dir` to `model.base_model`.
- [x] Respect `model.load_in_4bit` during prediction generation.
- [x] Train the first LoRA adapter on the current validated dataset.
- [x] Evaluate prompt-only Gemma vs fine-tuned Gemma on the same held-out set.
- [ ] Verify the Gemma 4 E2B/E4B QLoRA path end-to-end on the target 16 GB GPU and record peak VRAM.
- [ ] Test at least one alternative model config.
- [ ] Record training settings, GPU type, runtime, seed, dataset hash, and adapter output path.
