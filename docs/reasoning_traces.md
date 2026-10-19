# Reasoning trace experiments

Canonical traces store assistant reasoning separately from `content` and
`tool_calls`. The final answer still uses `<answer>...</answer>`. Reasoning is
optional at generation, training, and inference; changing one setting does not
automatically change the others.

## Teacher candidates

The initial shortlist follows the requested cost/intelligence tradeoff:

| Configuration | OpenRouter model | Reasoning settings |
| --- | --- | --- |
| `configs/teachers/teacher_deepseek_v4_1_flash.yaml` | `deepseek/deepseek-v4.1-flash` | Enabled, low effort |
| `configs/teachers/teacher_glm_5_3_flash.yaml` | `z-ai/glm-5.3-flash` | Included, low effort; internal thinking is mandatory |
| `configs/teachers/teacher_mimo_v2_6_flash.yaml` | `xiaomi/mimo-v2.6-flash` | Enabled; effort unspecified |

The public OpenRouter catalog checked on 2026-10-13 advertises `reasoning`,
`include_reasoning`, and `tools` for all three. This is a capability declaration,
not a measured guarantee that every tool-call and final-answer turn contains
readable reasoning. The pilot reports that coverage separately from correctness.
Model/provider prices change; use the recorded billed cost instead of fixed
prices or general intelligence scores to select the teacher for this dataset.
The initial low-effort settings favor shorter traces; Artificial Analysis scores
may use a different effort setting. Adjust effort when comparing quality, and
keep it fixed across runs being compared.

Sources: [DeepSeek V4.1 Flash](https://openrouter.ai/deepseek/deepseek-v4.1-flash),
[GLM 5.3 Flash](https://openrouter.ai/z-ai/glm-5.3-flash),
[MiMo V2.6 Flash](https://openrouter.ai/xiaomi/mimo-v2.6-flash), and the
[OpenRouter catalog](https://openrouter.ai/api/v1/models).

`reasoning.enabled` is the single switch for including reasoning in dataset traces.
`true` requests and retains returned reasoning. `false` requests its exclusion and
also strips any unexpectedly returned reasoning from accepted/rejected dataset
messages. This includes inline `<think>` blocks and opaque provider blocks.
It does not require a mandatory-thinking model to disable internal computation;
hidden reasoning can still be billed. Omission defaults to `false`.

`reasoning.effort` is optional. When omitted, no effort is sent and the selected
model/provider uses its default. `preflight` reports the catalog's default and
supported effort levels where available. The example configurations explicitly
use `low` for DeepSeek and GLM; MiMo leaves effort unspecified. The check's
`--reasoning-effort default` omits a configured effort without editing YAML.
An optional `reasoning.max_tokens` reasoning budget can be used when supported;
OpenRouter accepts effort or a token budget, not both. The top-level `max_tokens`
limits reasoning and visible output together on most providers.

The generation client sends exact JSON because the currently installed SDK omits
some reasoning fields during serialization. Readable text and summaries are
tagged separately. Encrypted blocks are never used as student training text.
Original signed blocks are replayed unchanged internally across tool turns,
even if saved dataset reasoning is disabled. Request journals retain the provider
response needed for resumable replay; the inclusion switch controls dataset
messages, not deletion of these API/debug artifacts.

## Cheap teacher capability checks

The three configurations are candidates to check, not measured recommendations.
Start with free catalog discovery (no API generation and no Neo4j):

```bash
docker/stack.sh main python scripts/agents/generate_traces.py preflight \
  --teacher-config configs/teachers/teacher_deepseek_v4_1_flash.yaml
```

This reports tools/reasoning parameters, reasoning defaults, and catalog pricing.
Advertised reasoning support and billed reasoning tokens do not prove that
readable reasoning text is exposed. The `check` command always runs exactly one
conversation using an existing dataset seed and live Neo4j:

```bash
docker/stack.sh main python scripts/agents/generate_traces.py check \
  --teacher-config configs/teachers/teacher_deepseek_v4_1_flash.yaml \
  --data-config configs/data.yaml --dataset-version SOURCE_VERSION \
  --max-tokens 2048 --max-requests 2 --max-cost-usd 0.10
```

There is no `--mode` or custom `--prompt` option. By default the command selects a
deterministic middle-sized, answerable Cypher QA seed and allows two requests:
tool generation and the final answer. Use `--seed-uid` to select a particular
QA/PDDL seed. If the model needs more attempts, the check stops at its explicit
request limit and saves the partial results; increase that limit only when you
want to inspect repair behavior. There are no automatic generation retries.

The check requests included reasoning by default, even if the YAML disables it.
`--no-include-reasoning` checks excluded output instead. `--reasoning-effort`
optionally overrides effort. A response truncated by the token limit is
inconclusive; increase `--max-tokens` if the report shows `finish_reason: length`.
No readable text in one successful response is reported as "not observed", not
proof that the model never exposes reasoning.

The command loads the selected scene into the configured Neo4j database, using
the same clearing/loading behavior as normal generation. It does not fork a
dataset, save training traces, or load a student tokenizer. It needs
`OPENROUTER_API_KEY`. Reports are saved under `outputs/teacher_checks/` unless
`--output` is supplied. Existing reports require `--force` to overwrite.
The default spending limit is $0.10; next-request cost estimates use advertised
catalog prices locally. Teacher requests send no provider routing parameters or
price filters. Actual provider prices can differ from the catalog estimate.
A budgeted check stops if a completed request has unknown billed cost.

The terminal report separates metrics, reasoning availability, answer validation,
and the generated example:

- Tables share the same title style and contain explanatory notes at the bottom,
  using the terminal's normal foreground color for readability.
- Costs use dollar units. Check costs retain small fractions of a dollar;
  estimated dataset costs use two decimal places (for example, `$0.70`).
- Durations use seconds, minutes, and hours (for example, `6.56 s` or
  `3 h 6 min 52 s`). Speed uses output tokens per second, rounded to two decimals.
- Tool-call and final-answer reasoning have separate rows. **Text** means readable
  reasoning; **Summary** means a readable summary, rather than the full trace;
  **Encrypted** and other opaque blocks are unusable as training text;
  **None** means no reasoning was returned. Readable-turn counts show partial
  coverage across retries, while saved-turn counts respect reasoning inclusion.
- Validation reports pass/fail/not-checked separately for Cypher execution,
  query-result or PDDL grounding correctness, answer format, and answer correctness.
- The outer **Generated text example** panel contains seed/scene/task information
  and panels showing the question, each generated Cypher call and its reasoning, tool
  errors or result counts, final-answer reasoning, the generated answer, and the
  expected answer. Expected-answer newlines are collapsed for display, while the
  saved JSON keeps the original formatting. The complete system/user prompt is omitted
  from terminal output.
  Reasoning availability and answer correctness do not certify that a rationale
  itself is sound; the displayed text supports manual inspection.

The JSON report retains numeric precision and full API responses; actual
model/provider and generation ID; finish reasons; reasoning type and coverage;
prompt, completion, reasoning and cached-token usage where returned; cost details;
request latency; and observed completion tokens/second. It also stores the selected
question, expected answer, model turns, and tool feedback for later inspection.
Throughput includes prompt processing and network latency; a non-streaming check
does not measure time to first token or pure decoder speed.

**Time in model requests** sums the duration from sending each API request to
receiving its full response, including network and provider waiting time.
**Example generation time** measures the whole conversation: both API requests,
Cypher execution, and answer validation. Model discovery, loading the Neo4j scene,
and creating the validator happen before this timer starts and are excluded.
For example, a measured conversation duration of `3.128408 s` across `1,660`
sequential traces gives about `1 h 26 min 33 s`; this estimate uses unrounded
timings rather than multiplying the rounded `3.13 s` shown in the table.

Dataset estimates multiply the measured conversation cost and sequential time
by the number of source seeds and configured `traces_per_seed`. Override the seed
count with `--dataset-size` and copies with `--traces-per-seed`. Actual billed cost
is preferred; missing cost falls back to a clearly labeled catalog estimate when
possible. Reasoning tokens are already included in completion billing.
Interrupted and token-truncated checks do not extrapolate full conversation costs.
One example cannot estimate acceptance coverage across all question types, scene
loading overhead, rate limits, or parallel generation speed; use the pilot when
you need those broader quality measurements.

API references: [reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
and [response usage](https://openrouter.ai/docs/api/reference/overview).

## Synthetic reasoning

Copy `configs/teachers/teacher_synthetic.yaml` and enable reasoning:

```yaml
provider: synthetic
traces_per_seed: 1
max_tool_calls: 2
inject_controlled_repair: true
reasoning:
  enabled: true
  max_sentences: null
```

`null` includes every relevant decision unit; there is no default sentence cap.
An integer greater than or equal to zero limits each assistant turn independently.
`max_tool_sentences` and `max_final_sentences` override the shared cap for their
respective stages. An override of `null` restores unlimited sentences for that
stage, and zero emits no reasoning text while retaining selection diagnostics.

Synthetic reasoning renderer version 3 uses validated retrieval operations and the
existing PDDL goal tree. Retrieval declarations remain at semantics version 1.
Sentence triggers do not inspect question wording, tags, or
`query_type`. The same operation rules handle new combinations of supported tasks.
Before-tool reasoning never reads expected answers, expected grounding, or tool rows.
Final reasoning uses observed rows and public grounding roles only.

The two stages describe different decisions:

- Before-tool reasoning describes retrieval constraints, containment/connectivity,
  optional matches, grouping, ranking, ties and repairs. It omits routine entity
  matching, projection narration and all PDDL predicates, action meanings and goal
  composition. Simple symbol/class lookups and plain counts can have no reasoning.
  Query filters retain their own logical operators when they restrict candidates.
- Final PDDL reasoning describes predicate arguments and the complete goal structure
  using task roles instead of repeating retrieved symbols. `ANY` denotes an `or`
  expansion across candidates; `ALL` denotes an `and` expansion. Explicit goal
  operators use `AND(...)`, `OR(...)` and `NOT(...)`. For example,
  `AND(ALL safe(chair objects), ANY in-region(office rooms)).` Distinct target roles,
  placement argument order and negation scope are preserved. Singleton expansions
  need no `ANY`/`ALL` wrapper, matching the grounded PDDL answer.
- Final QA reasoning omits result values, column mapping and routine zero-count
  narration. Short syntax reminders remain: `SLDP integer.`,
  `SLDP POINT(x y z); no commas.`, or `SLDP set of symbol pairs; preserve row pairs.`
  PDDL finals retain `PDDL goal expression.` No stage repeats answer-tag instructions.
- Unlisted-class notices remain concise: `Unlisted class laundry; query for instances.`
  An absent label never establishes that the scene has no matching instances.

Related retrieval decisions share one unit. Distance computation has no standalone
sentence; its metric appears in the relevant bound or ranking. A radius followed by
nearest selection can read `center distance(...) <= 5; Nearest qualifying candidates.`
Counts and extrema already explained by ranking or tie selection are not narrated
again. More complex goals retain their complete logical structure in one unit.

Retrieval declarations live in
`configs/question_templates/synthetic_reasoning/cypher_qa_3dsg.yaml` and
`configs/question_templates/synthetic_reasoning/cypher_pddl_3dsg.yaml`. Each file has one global
`version` and a `templates` mapping keyed by the existing `query_type` names.
Question catalogs retain question, query and sampling fields. Validation contracts
and PDDL goal trees live under `configs/question_templates/validation/`; the shared
loader supplies those goals to reasoning. See [Cypher and grounding validation](cypher_grounding_validation.md).
Question catalogs have no `template_id`, `oracle_type` or `semantics` fields. Declarations are
compiled from Cypher by a parser for the supported read-only subset, retaining
bindings, layers, directed containment, bounded connectivity, logical filters,
computations, grouping, projections, order and limits. PDDL intent continues to come
from
`goal`; it is not inferred from the retrieval query. This avoids maintaining rules
for each question name and avoids duplicating the PDDL goal specification.

These files describe complete retrieval queries, not the emitted sentence types.
Keep `match`, projection and computation operations when their narration is removed:
they supply bindings, class/layer information, unlisted-class checks, selection
metrics and final PDDL target roles. The declarations remain private metadata;
only renderer-selected sentences enter supervised reasoning. Renderer-only
sentence changes require no declaration update.

After changing a query, explicitly refresh its declarations and review the diff:

```bash
python scripts/agents/audit_synthetic_reasoning.py compile --write
python scripts/agents/audit_synthetic_reasoning.py audit
```

The compiler writes only the separate reasoning files. It never rewrites question
catalogs or prompts. Adding or renaming a question type, or changing its query,
requires recompilation to update the mapping and operations; it needs no renderer
change for supported operations. Changing tags, question variants or sampling
settings requires no reasoning configuration change. `query_type` also remains the
existing dispatch key for scene-based gold answers and configuration lookup.
Result interpretation and comparison use the resolved validation contract. There is
no additional task identifier or oracle dispatch field; template provenance
identifies the exact source query and explicit bindings. An operation outside the supported
vocabulary requires parser/rendering support.
The scene-based gold-answer oracle is a separate component; new question
families also need oracle support, even when their reasoning operations are reused.

Question generation retains explicit sampled bindings, template identity and
the resolved contract without loading these files. Synthetic teacher generation with reasoning enabled verifies the separate
declarations and attaches resolved operations, goal structure, public grounding
fields and a fingerprint to `metadata.reasoning_context`. Real teachers and
synthetic generation with reasoning disabled skip this configuration entirely.
Metadata is private and adds no text to prompts. Reasoning configuration and source
fingerprints enter synthetic reasoning run identity, preventing stale resumable runs.
The synthetic teacher's optional `reasoning.semantics_paths` selects custom files,
in the same order as `question_sets`; defaults use each question file's basename
under `configs/question_templates/synthetic_reasoning/`. The offline CLI accepts
the same override through `--semantics`, paired with `--templates`.

Validation normalizes whitespace, keyword case, simple local-variable renames and
independent conjunction order. It is a structural consistency check, not a general
Cypher equivalence proof. Other rewrites may require recompilation. Unknown
functions, unbounded traversals, arbitrary procedures and unsupported list
comprehensions receive explicit diagnostics instead of partial explanations.

| Priority | Stage and rule |
| --- | --- |
| 0 | Observed repair; refusal; empty outcome; missing required grounding or unavailable alternatives. |
| 10 | Final PDDL composition, including predicates, quantification, negation and destination roles. |
| 20 | Retrieval constraints, containment/connectivity, optional matches, bounds, aggregation, extrema and selection. |
| 30 | Unlisted-class notices and final output syntax. |

Candidates are selected by priority and stable rule ID, then rendered in task order.
Deduplication includes scope. A PDDL logical expression is one complete sentence
unit, so caps cannot keep existential choice while dropping its enclosing negation.
Every generated turn records `cap`, `candidate_count`, `selected_rules`,
`omitted_rules`, `critical_omitted_rules`, `rendered_rules`, and renderer version.
A small cap may omit essential decisions; diagnostics expose those omissions.
Unlimited selection remains the default. Reasoning remains opt-in.

Final explanations specify concise SLDP shapes, preserve pair rows, and inspect
the goal tree before dropping unavailable alternatives. Empty answers retain their
container syntax (`<>` or `[]`); zero counts retain ordinary integer syntax.
An unavailable required conjunct produces an empty answer.
Repairs retain their original masked faulty turn and identify the observed query
correction where possible.

### Offline compilation and audit

Generation retains explicit `metadata.template_bindings` and template identity.
Synthetic teachers with reasoning enabled use those bindings and the selected
semantics catalog to build verified retrieval/goal snapshots. Context schema
version 2 stores contract role definitions; the separate operation declarations
remain at version 1. Real teachers and
reasoning-disabled synthetic teachers do not load the synthetic catalog. Unknown
or contradictory provenance requires regeneration.

The audit works on copies of current seeds and does not call a model or load/clear
Neo4j. It verifies template/query agreement, previews both stages using saved tool
feedback, and reports coverage, lengths and cap omissions. Outputs must be new files:

```bash
python scripts/agents/audit_synthetic_reasoning.py audit \
  --seeds data/datasets/dataset_demo/SOURCE_VERSION/cypher_seed_records.jsonl \
  --traces data/datasets/dataset_demo/SOURCE_VERSION/traces/teacher_traces.jsonl \
  --output /tmp/reasoning-audit.json
```

Use `--tokenizer /local/tokenizer/directory` for exact reasoning-token counts
without downloading a model. Word counts alone do not establish training-length
usability. Compile declarations into their separate synthetic configuration;
auditing current records never imports or rewrites unsupported old formats.

### Matched experiments

Generate a fresh validated dataset with the existing prompts and enabled synthetic
reasoning, then split it once. Use the existing matched-retained-UID mechanism for
reasoning-on/off training, ensuring that length filtering retains identical examples.
Saved traces remain unchanged. To compare
versions 2 and 3, retain those traces and regenerate with the current renderer.
For a version-1 comparison, preserve a separately generated version-1 corpus from
the earlier code and intersect retained UIDs across the corpora. Align prompts,
queries, answers, splits, tokenizer, optimizer settings and training seeds; varying
prompt versions would confound a reasoning-only comparison.

Compare executable/query-result correctness, final-answer correctness, performance
by semantic operation, reasoning length and cap omissions. An offline coverage
report does not establish a student accuracy improvement. Full model training and
fresh Neo4j execution are separate validations after dataset regeneration.

The synthetic teacher retains controlled repair supervision. The injected faulty
assistant turn and its error remain masked; the corrected call explains the fix.
Real teachers always skip injected repairs, even when old seeds contain them.
`inject_controlled_repair` is a synthetic-only option; an enabled value on another
provider is ignored with a warning.
They retain the same question and seed identity, and preserve natural errors,
feedback, corrections, and reasoning in their resulting conversations.

## Reproducible cost and coverage pilot

Run this command for each candidate, keeping the same input dataset and sample
settings. Generation forks an existing dataset as usual, so retain the dataset
version printed by each run.

```bash
docker/stack.sh main python scripts/agents/generate_traces.py generate \
  --data-config configs/data.yaml \
  --dataset-version SOURCE_VERSION \
  --teacher-config configs/teachers/teacher_deepseek_v4_1_flash.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --pilot --pilot-per-task-type 2 --pilot-extra-per-tag 2 \
  --sample-seed 42 --max-cost-usd 3 --preflight
```

Sampling covers every task type and every represented scene within that type.
The per-type count is a minimum: scene coverage may increase it. Extra examples
cover invalid inputs, negation, conjunction, disjunction, universal targets,
ordering, and place questions. `seed_manifest.yaml` records the exact selected
UIDs and source fingerprint. Compare that manifest across models.

The first small pilot is also the capability smoke test: inspect tool-stage and
final-stage readable reasoning coverage before committing to full generation.
To use fewer examples initially, set `--pilot-per-task-type 1
--pilot-extra-per-tag 0`; scene coverage still applies.

Each request records its model, provider, finish reason, latency, usage, and billed
cost in `requests.jsonl`, including requests from rejected or unfinished
conversations. Reasoning tokens are a subset of completion tokens and are not
charged a second time in the report. When a spend limit is supplied, the client
loads catalog pricing automatically and estimates the next request's cost before
sending it, using the prompt byte length and maximum completion tokens. This
estimate does not enforce a provider price ceiling, so actual costs can exceed
the spend limit. Missing catalog prices or billed costs stop further budgeted
requests.

`attempts.jsonl` checkpoints completed outcomes. On resume, saved API responses
are replayed and database queries are rerun, avoiding duplicate charges:

```bash
docker/stack.sh main python scripts/agents/generate_traces.py generate \
  --data-config configs/data.yaml --dataset-version PILOT_VERSION \
  --teacher-config configs/teachers/teacher_deepseek_v4_1_flash.yaml \
  --pilot --pilot-per-task-type 2 --pilot-extra-per-tag 2 \
  --sample-seed 42 --max-cost-usd 3 --resume
```

Keep the teacher, tokenizer, schema, prompts, sampling settings, and data unchanged.
The spend limit can be increased on resume. If interruption occurred while a
request was in flight and its response was not saved, `pending_request.json`
blocks automatic retry: reconcile that request with OpenRouter first, since its
cost and response cannot be inferred safely. A stopped pilot cannot be split as
a complete training dataset. Run full generation without `--pilot` after selecting
the teacher.

`pilot_report.yaml` reports correctness acceptance, readable reasoning at both
supervised stages, training-length usability, rejection reasons, and coverage
by scene, task type, and tags. It distinguishes summaries from raw reasoning.
Full-dataset cost is estimated from scene/family/task-type weighted mean
conversation costs, including rejected conversations. Unsampled strata or missing
costs produce an incomplete estimate. A sampling interval is reported only when
every stratum has enough observations; these small pilots remain exploratory.
API credentials and live Neo4j execution are required to measure real coverage.

## Matched student experiments

Split a complete trace dataset once. Reasoning is retained in canonical and
rendered records; use the same split files for both training variants:

```bash
docker/stack.sh main python scripts/training/train_agent_lora.py \
  --data-config configs/data.yaml --dataset-version DATASET_VERSION \
  --student-config configs/students/student_qwen3.5_4b.yaml --include-reasoning

docker/stack.sh main python scripts/training/train_agent_lora.py \
  --data-config configs/data.yaml --dataset-version DATASET_VERSION \
  --student-config configs/students/student_qwen3.5_4b.yaml --no-include-reasoning
```

Alternatively set `training.include_reasoning` in the student config. Set
`training.retained_uids_path` to that completed run's
`traces/matched_retained_uids.json` for both variants to use the same examples
after length filtering. This list is specific to the selected tokenizer and
training length limit. It does not alter split assignments. Overlength traces
are excluded as before; reasoning is never silently truncated to fit.

Qwen reasoning is rendered through its native `reasoning_content` field. During
training, the historical-turn gate is adapted so XML tool feedback
cannot remove prior reasoning or break assistant-only token masks. Models with
no native reasoning field, including the saved Granite template, use portable
`<think>` blocks. Both tool-stage and final-stage reasoning receive assistant loss.

Inspect the actual training representation and masks with:

```bash
docker/stack.sh main python scripts/training/inspect_training_dataset.py \
  --data-config configs/data.yaml --dataset-version DATASET_VERSION \
  --student-config configs/students/student_qwen3.5_4b.yaml --include-reasoning
```

Inference defaults to `inference.enable_thinking: false`. Override it with
`--enable-thinking` or `--no-enable-thinking` on `evaluate_student.py`. Use both
modes for each trained variant and compare end-to-end correctness, query retries,
PDDL grounding, generated tokens, latency, and length failures. Transformers and
Ollama retain reasoning separately from executable actions. Unfinished think
blocks are not executed. Ollama's separate `thinking` field is retained in debug
reports and replayed in reasoning-enabled conversations. Native Ollama thinking
support still depends on the exported model/server; portable reasoning support
in Transformers does not imply that every Ollama model accepts `think: true`.

For external benchmark sweeps, set `parameters.reasoning.mode: enabled` or
`disabled` in a copy of `examples/herables_benchmark/config/model_lists/students.yaml`.
Benchmark settings are independent of this project's evaluation CLI defaults.

`optimizations.repair_invalid_final_answers` in the teacher YAML enables a
post-generation optimization pass. It repairs eligible SLDP format-only rejects
while preserving every tool call, its feedback, and readable, summary, inline or
opaque reasoning. Inline thinking in the final message is retained in its
separate reasoning field. Repaired traces are explicitly marked and count as
accepted in the finalized generation outputs. The progress bar, summary, run
metadata and pilot report show how many repairs contributed to acceptance;
original rejection evidence remains in the attempt journal. Splitting consumes
these finalized outputs without applying repairs.

`optimizations.tuning_prompt` adds teacher-specific guidance to each outbound
request, including subsequent turns after tool feedback. The text is appended
to a copy of the user message and excluded from saved conversation messages.
It remains in teacher run metadata for reproducibility. Synthetic teachers
ignore both optimizations with a warning when enabled.
