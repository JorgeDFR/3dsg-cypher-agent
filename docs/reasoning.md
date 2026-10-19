# Reasoning Experiments

Reasoning is optional in teacher traces, student training, and inference.
Configure each stage separately. Enabling one does not enable the others.

## Remote teacher reasoning

Set reasoning in the teacher YAML:

```yaml
reasoning:
  enabled: true
  effort: low
```

| Setting | Behavior |
| --- | --- |
| `enabled: true` | Request and retain returned reasoning |
| `enabled: false` | Exclude reasoning from saved dataset messages |
| `effort` | Request a supported effort level, or omit for the provider default |
| `max_tokens` | Optional reasoning budget, when supported by the provider |

Use either effort or a reasoning-token budget. The teacher's top-level
`max_tokens` limits the whole response where the provider counts reasoning and
visible output together.

Disabling retained reasoning does not guarantee that the model stops internal
reasoning or its billing. API journals can still contain provider responses.
Encrypted reasoning is not used as student training text.

Use [Teacher checks and pilots](teacher_checks.md) to inspect readable reasoning
at tool-call and final-answer stages before generating a full dataset.

## Synthetic reasoning

The synthetic teacher describes retrieval decisions and final-answer structure.
Set these options in a copy of `configs/teachers/teacher_synthetic.yaml`:

```yaml
reasoning:
  enabled: true
  max_sentences: null
```

| Setting | Purpose |
| --- | --- |
| `max_sentences` | Shared cap for each assistant turn |
| `max_tool_sentences` | Override the cap before tool calls |
| `max_final_sentences` | Override the cap before final answers |

`null` keeps all relevant decision units. Zero emits no reasoning text.
A small cap can omit important decisions. Inspect cap omissions in the audit
before using capped traces for training.

Before-tool reasoning describes retrieval constraints without using expected
answers. Final reasoning uses observed results. PDDL explanations retain goal
composition, argument roles, and negation. Simple lookups may have no reasoning.

### Update declarations after query changes

Synthetic retrieval declarations live in
`configs/question_templates/synthetic_reasoning/`. Validation rules and PDDL
goals belong in [validation contracts](validation_contracts.md).

After adding or renaming a question type, or changing its Cypher, refresh and
check the declarations:

```bash
docker/stack.sh main python scripts/agents/audit_synthetic_reasoning.py compile --write
docker/stack.sh main python scripts/agents/audit_synthetic_reasoning.py audit
```

Compilation updates only the separate reasoning files. Changes to question
wording or sampling do not require recompilation. Unsupported query operations
need code support before they can generate verified reasoning.

### Audit saved traces

```bash
docker/stack.sh main python scripts/agents/audit_synthetic_reasoning.py audit \
  --seeds data/datasets/dataset_demo/SOURCE_VERSION/cypher_seed_records.jsonl \
  --traces data/datasets/dataset_demo/SOURCE_VERSION/traces/teacher_traces.jsonl \
  --output /tmp/reasoning-audit.json
```

The audit checks query agreement, reasoning coverage, length, and cap omissions.
It uses saved evidence without model requests or database loading. Outputs must
be new files. Add `--tokenizer /local/tokenizer/directory` for exact token counts.

## Compare student training with and without reasoning

Generate a complete dataset with reasoning, then split it once. Use the same
split files for both training variants:

```bash
docker/stack.sh main python scripts/training/train_agent_lora.py \
  --data-config configs/data.yaml --dataset-version DATASET_VERSION \
  --student-config configs/students/student_qwen3.5_4b.yaml --include-reasoning

docker/stack.sh main python scripts/training/train_agent_lora.py \
  --data-config configs/data.yaml --dataset-version DATASET_VERSION \
  --student-config configs/students/student_qwen3.5_4b.yaml --no-include-reasoning
```

The equivalent setting is `training.include_reasoning` in the student YAML.
Set `training.retained_uids_path` to the first completed run's
`traces/matched_retained_uids.json` for both variants. This keeps the examples
identical after length filtering. Keep the tokenizer and length limit fixed.

Inspect the supervised text with
[training dataset inspection](training.md#inspect-the-rendered-dataset).
Compare correctness, grounding, retries, generated tokens, latency, and length
failures. Coverage alone does not establish an accuracy improvement.

## Compare inference modes

Inference defaults to `inference.enable_thinking: false` in the student YAML.
Pass `--enable-thinking` or `--no-enable-thinking` to `evaluate_student.py`
to compare both modes for each trained variant.

Transformers and Ollama retain reasoning separately from executable tool calls.
Ollama thinking support depends on the exported model and server.

For external sweeps, set `parameters.reasoning.mode` to `enabled` or `disabled`
in a copy of
`examples/herables_benchmark/config/model_lists/students.yaml`.
These settings are independent of the project's evaluation defaults.
