# TODO

This file tracks the cheap, practical next steps for turning the repo into a reproducible
text-to-Cypher fine-tuning experiment for Heracles/Hydra 3D Scene Graphs.

## Dataset

- [ ] Load 3-5 Heracles scene graphs into Neo4j and assign stable `scene_id` values.
- [ ] Export real object classes, room symbols, place symbols, mesh-place symbols, and useful point/bounding-box anchors from each scene.
- [ ] Replace the placeholder values in `configs/templates_3dsg.yaml` with scene-derived values.
- [ ] Expand the template library to roughly 50 Cypher templates across:
  - [ ] object lookup
  - [ ] class inventory
  - [ ] counts and aggregation
  - [ ] hierarchy and transitive `CONTAINS*`
  - [ ] spatial distance
  - [ ] bounding boxes
  - [ ] nearest-neighbor queries
  - [ ] connectivity/hop queries
  - [ ] relationship reasoning
  - [ ] read-only refusal/safety examples
- [ ] Generate 10-20 question paraphrases per template.
- [ ] Add `expected_result_hash` for gold queries by executing them against Neo4j.
- [ ] Split train/eval/test by `scene_id`, not only by row, once multiple scenes are available.

## Validation And Evaluation

- [ ] Add result-equivalence evaluation between gold Cypher and predicted Cypher.
- [ ] Normalize unordered result rows before hashing/comparison.
- [ ] Add numeric tolerance for point coordinates and distance outputs.
- [ ] Report metrics by `query_type` and `tags`.
- [ ] Track invalid schema references separately from syntax errors.
- [ ] Add a guardrail check for overly broad queries, such as unbounded traversals or missing `LIMIT` in nearest-neighbor patterns.
- [ ] Add a read-only Neo4j user for live validation and prediction evaluation.

## Fine-Tuning

- [ ] Confirm the exact Gemma 4 checkpoint name/path and update `configs/model_gemma4.yaml`.
- [ ] Run a tiny overfit test on 20-50 examples to verify the training loop.
- [ ] Train the first LoRA adapter on 500-1,000 validated examples.
- [ ] Evaluate prompt-only Gemma 4 vs fine-tuned Gemma 4 on the same held-out scene.
- [ ] Add at least one alternative model config, such as Qwen Coder or Llama, without changing dataset code.
- [ ] Record training settings, GPU type, runtime, and adapter output path for reproducibility.

## Paper Artifacts

- [ ] Create tables for syntax-valid, executable, result-equivalent, timeout, and safety-pass rates.
- [ ] Create per-query-family plots.
- [ ] Save representative failure cases for error analysis.
- [ ] Document examples where fine-tuning helps and where prompt-only performance is already sufficient.
- [ ] Preserve the exact dataset split files used for each reported experiment.

