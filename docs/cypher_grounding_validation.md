# Cypher results and grounding validation

Validation contracts describe what a query must return and how to compare its
result and final answer. They are shared by seed generation, synthetic and real
teachers, standalone Cypher evaluation, student evaluation, and saved-trace audits.
Reasoning can be enabled or disabled independently.

## Configuration ownership and discovery

```text
configs/question_templates/
  cypher_qa_3dsg.yaml
  cypher_pddl_3dsg.yaml
  synthetic_reasoning/             # retrieval operations for synthetic narration
  validation/
    defaults.yaml
    cypher_qa_3dsg.yaml             # 52 contracts
    cypher_pddl_3dsg.yaml           # 39 contracts, including PDDL goal trees
```

Main question catalogs own wording, variables, gold Cypher, tags, layer eligibility,
target counts, sampling groups, and sampling/repair/size limits. They contain no
`solution_formats`, `grounding`, `goal`, `answer_validation`, or `read_only` fields.
Sampling limits control dataset composition; `result.row_count` controls validation.

A validation catalog has `version: 1` and `templates`, keyed by `query_type`.
`defaults.yaml` has `version: 1` and `defaults`. Configuration precedence is shared
defaults followed by recursive per-template overrides. Lists are replaced.
Unknown fields, duplicate YAML keys/types, obsolete or missing entries, incompatible
shapes, invalid goals, and references to nonexistent variable roots are errors.

By default the loader finds `validation/<question-catalog-basename>` beside the
question catalog and `defaults.yaml` beside that validation catalog. A question
set in `configs/data.yaml` can override either path:

```yaml
question_sets:
  - name: cypher_qa
    question_templates_path: cypher_qa_3dsg.yaml
    experiment_profile_path: cypher_qa_template.yaml
    validation_config_path: configs/question_templates/validation/cypher_qa_3dsg.yaml
    validation_defaults_path: configs/question_templates/validation/defaults.yaml
```

Paths use the existing project/config resolution helpers. The loader's private
composed view supplies compatibility fields to seed/oracle/reasoning builders;
there is one declaration on disk. Real teachers never need synthetic reasoning
files. A standalone query-only reasoning scratch file can still be compiled
without a validation sidecar; production generation requires one.

## Contracts and controlled bindings

A count contract:

```yaml
version: 1
templates:
  object_class_count:
    result:
      kind: scalar
      type: integer
      row_count: {min: 1, max: 1}
    answer:
      formats: [int]
      empty: '0'
```

`count(o)` must return one integer, including zero. No rows, null, a boolean,
fractional count, or multiple values are shape errors. Column names are cosmetic:
`RETURN count(o) AS total` with `[{total: 3}]` decodes to `3`.

A point set contract:

```yaml
object_class_centers:
  result:
    kind: set
    type: point
    property: center
    layers: [Object]
    accepted_forms: [scalar_columns, collected_columns, nodes]
    comparison: {point_abs_tolerance: 0.01}
  answer:
    formats: ["set[point]"]
    empty: '<>'
    comparison: {point_abs_tolerance: 0.01}
```

Points accept three finite coordinates, Neo4j points, `{x, y, z}` objects, or
`POINT(x y z)` text. A coordinate triple is one point; a collection of triples is
several points. Floating-point result rendering preserves precision.

A PDDL contract with two Object roles:

```yaml
pddl_visit_object_and_end_at_object:
  result:
    kind: role_sets
    roles:
      waypoint:
        type: node_symbol
        layers: [Object]
        property: nodeSymbol
        oracle_source: waypoint
        origin: {nodeSymbol: {binding: waypoint_object}}
        cardinality: set
      destination:
        type: node_symbol
        layers: [Object]
        property: nodeSymbol
        oracle_source: destination
        origin: {nodeSymbol: {binding: destination_object}}
        cardinality: set
  goal:
    operator: and
    clauses:
      - fact: visited-object
        arguments: [{result: waypoint}]
        multiple: or
      - fact: at-object
        arguments: [{result: destination}]
        multiple: or
  answer:
    empty: ''
    comparison: {mode: pddl_equivalence}
```

`{binding: name}` resolves against sampled variables, including dotted names such
as `case.layer`. Origins describe controlled query inputs, never expected output
IDs. `oracle_source` selects a field from the trusted scene oracle during seed
construction; it is not a model-response column hint.

For example, with `waypoint_object: O1` and `destination_object: O2`, a teacher may return:

```cypher
MATCH (a:Object {nodeSymbol: "O1"}), (b:Object {nodeSymbol: "O2"})
RETURN b.nodeSymbol AS finish, collect(a.nodeSymbol) AS stops
```

`[{finish: "O2", stops: ["O1"]}]` maps to `destination` and `waypoint` from
projection provenance. Changing expected IDs changes comparison, never decoding.
Explicit semantic output names such as `waypoint` are authoritative: returning
`b.nodeSymbol AS waypoint` conflicts with its declared origin and cannot be
silently swapped into a pass. Two renamed Object columns with neither provenance
nor an adapter remain `mapping_unresolved` when assignments differ.

Independent roles are different from tuple relations. To preserve Object/Place
associations across rows:

```yaml
result:
  kind: tuples
  roles:
    object: {type: node_symbol, layers: [Object], property: nodeSymbol}
    place: {type: node_symbol, layers: [Place], property: nodeSymbol}
answer:
  formats: ["set[list[str]]"]
  empty: '<>'
```

`(O1, p1), (O2, p2)` differs from `(O1, p2), (O2, p1)`. Tuple field order follows
the declared role order. `unordered_tuple: true` is only appropriate for symmetric
pairs; existing QA proximity pairs use it. Goal rendering currently requires
`role_sets`; tuple relations are available for QA/result comparison.

## Schema reference

All objects reject unknown keys. Version 1 has these fields:

| Object | Fields and defaults |
| --- | --- |
| Contract | `version: 1`; required `result`; `query_policy`, `answer`, `semantic_checks` use defaults; optional `goal` |
| `query_policy` | `read_only: true` cannot be weakened; optional `refusal` literal |
| `result` | Required `kind`: `scalar`, `set`, `multiset`, `list`, `tuples`, `role_sets`, or `flexible` |
| Simple result type | `type: string`, `layers: []`, `property: null`, `origin: {}`; optional `element` gives the full value specification below |
| Structured result | `roles: {}` maps role names to value specifications; required for `tuples`/`role_sets` |
| Representations | `accepted_forms: [scalar_columns, collected_columns, nodes]`; add `maps` for declared structured layouts |
| Mapping | `mapping: provenance_or_unique_type` or `explicit`; `ambiguity: unresolved`; no expected-value or implicit positional fallback |
| Extra output | `extra_columns: inferred` permits typed diagnostics after answer mapping; `reject` forbids extras; `declared` permits only named, typed `auxiliary` columns |
| Shape | `row_count: {min: 0, max: null}`; `unordered_tuple: false`; `symbol_case_sensitive: true` |
| Value specification | `type`: `integer`, `number`, `string`, `node_symbol`, `point`; `layers: []`; `property: null`; `origin: {}`; `oracle_source: value` |
| Value collections | `cardinality: set` or `list`/`multiset` for role sets; `nulls: reject` or explicit `empty_set`; `dimensions: 3` |
| Value adapters | Optional `column` **or** zero-based `position`; `path: []` selects explicit nested map keys/list indices |
| `answer` | `formats: []`, `empty: ''`, `answer_tags: exactly_one`, `comparison` |
| Comparison | `mode: exact`, `sldp`, `pddl_equivalence`, or `top_k_with_ties`; `numeric_abs_tolerance: 0`; `point_abs_tolerance: 0.01`; optional `limit`, `scores`, `score_source` |
| `semantic_checks` | `mode: supported_fragment` (default) or `unverified`; runtime query-analysis policy only |

`result.comparison` controls execution-vs-oracle comparison. `answer.comparison`
controls final-answer-vs-oracle comparison. Configure both when both need the same
nondefault policy. Final grounding checks compare against what the tool actually
returned and do not substitute a different oracle-approved tie answer. They retain
the answer's numeric/point tolerances and omit ranking-only `limit`, `scores`, and
`score_source` settings from that comparison.

Supported answer formats use `str`, `int`, `float`, `point`, `bool`, and nested
`set[...]`/`list[...]` notation. Empty representations must satisfy the declared
format; empty text is allowed only when configured. Strings and node IDs compare
case-sensitively: `p119` and `P119` are different. Set comparisons deduplicate;
lists retain order; multisets retain multiplicity. Numeric/point matching uses
one-to-one matching, so one approximate match cannot satisfy two expected items.

`flexible` supports legacy scalar-or-set answer shapes: one value is scalar,
otherwise values form a set. Prefer an explicit shape for new templates.

For `mapping: explicit`, every role (or the simple result's `element`) requires a
column or position. Position adapters use preserved driver column order. A
controlled nested map layout can select multiple fields from one column:

```yaml
result:
  kind: role_sets
  mapping: explicit
  accepted_forms: [maps, scalar_columns]
  roles:
    object: {type: node_symbol, layers: [Object], column: payload, path: [target]}
    room: {type: node_symbol, layers: [Room], column: payload, path: [region]}
```

Full Neo4j nodes retain labels and properties in a JSON-safe typed envelope before
`record.data()` can erase labels. Declared properties are extracted from nodes
and property maps. Scalar columns and `collect` results use the same value rules.
Invalid collection members are errors; they are never filtered away. Configured
auxiliary columns are validated too.

With `extra_columns: inferred`, known property and expression types identify
diagnostics without an alias allowlist. For example, `RETURN o.nodeSymbol AS id,
point.distance(o.center, p.center) AS d` can decode just the object symbols while
validating `d` as finite numbers. Types survive `WITH` and `min`/`max` aliases.
Unknown auxiliary expressions still require a typed `auxiliary` declaration.
All plausible answer mappings are considered before extras are excluded: two
equally plausible symbol columns with different values remain unresolved.
Empty rows use projection types to distinguish answer columns from diagnostics.

The default applies to newly generated contracts. Saved seeds and traces retain
their explicit policies, including the previous `extra_columns: reject` default.
Regenerate raw questions and prepared seeds in a new dataset version to adopt
the new default before generating teacher traces. Forking an existing dataset
for another teacher copies its saved contracts unchanged. Historical replay with
the new policy must explicitly apply it to copies; it does not rewrite old grades.

PDDL goals accept `fact`, `arguments` (`result` or `literal`), `multiple: and|or`,
and optional `negate_each`; compound goals accept `operator: and|or|not` and
`clauses`. Role and predicate types are checked. Missing alternatives can be
removed from OR; a missing required AND or NOT operand leaves no satisfiable goal.
Gold domain/shape errors are `invalid_expected_solution`, not model failures.

Ranking uses `mode: top_k_with_ties` and a controlled limit. For the current class
ranking template, `score_source: {kind: property_counts, layer: Object, property:
class}` freezes scene counts in each record's `scores`. The generic source also
supports other layers and `property: nodeSymbol`. Answers must contain exactly
`min(limit, number_of_candidates)` unique entries, include all entries strictly
above the cutoff, and choose the remainder among cutoff ties. No new query-type
branch is needed to configure that source.

Refusal templates declare `query_policy.refusal: WRITE_QUERY_NOT_ALLOWED`.
Existing common schema/write/function/resource safety checks still apply to
ordinary queries. Fixed variable-length paths `*2`, `*2..2`, and `*..2` are bounded;
`*`, `*2..`, and `*..` are rejected. Path checks mask quoted strings, comments,
and backtick identifiers.

## Interpretation, correctness, and evidence

Decoding takes the actual query, original rows/column metadata, and resolved
contract. Expected solutions are not decoder inputs. Projection analysis tracks
supported MATCH/WHERE constraints, WITH, RETURN, maps, `collect`, and UNWIND.
Node identity survives aliases so same-node symbol, class, native node properties,
and `labels(n)` observations can establish a role. Equality and literal `IN`
constraints can describe a pooled source. For `role_sets`, a shared symbol or
native node column can be partitioned by those public origin constraints; tuple
relations retain their original row associations. Every pooled row must belong
to exactly one assigned role, and invalid values are never filtered away.
An absent pooled role needs a complete source covering its constraints; `LIMIT`
prevents that inference. Collected parallel arrays are not implicitly zipped.

Top-level `UNION`/`UNION ALL` branches retain separate provenance. Actual symbol
layers, explicit IDs, and same-node observations can distinguish branches. When
several branches remain possible, they must agree on a role's provenance before
that row can be assigned. A simple typed set can still decode when branch
identity has no effect on its interpretation. Keywords inside literals or
comments do not create branches. These rules never consult expected answers,
expected grounding sets, or the generated final answer to choose a mapping.

Unsupported syntax can still produce an unambiguous typed scalar; it cannot
supply missing provenance needed to distinguish same-type roles. At most 256
assignments are considered; more remain unresolved.

Reports retain original rows and aliases, canonical actual values, chosen
bindings, `auxiliary_columns`, partitioned `role_rows` (zero-based indices), shape
issues, and missing/extra role values. Checks have separate
outcomes:

- `result_validation`: `pass`, `mismatch`, `invalid_result_shape`,
  `mapping_unresolved`, `invalid_expected_solution`, or `not_checked`.
- Final-answer format/correctness: exactly one complete `<answer>` block and the
  configured comparison. Multiple or nested blocks fail.
- `grounding_check` / `final_answer_grounded`: whether the final answer matches
  the actual returned result, including the goal rendered from actual PDDL roles.
- `semantic_check`: `equivalent_in_supported_fragment`, `semantics_unverified`,
  for runtime query analysis.

Result equality means equality on the evaluated scene. Conservative AST evidence
supports variable renaming, cosmetic return aliases, and identical fixed path
bounds while retaining clause boundaries, optionality, direction, grouping,
ordering and limits. Different ASTs remain unverified; they do not prove an error.
Legacy `potential_hidden_false_positive` diagnostics remain readable for old
reports; new contract results do not use them as correctness decisions.

`semantic_checks.mode` controls this runtime diagnostic: `supported_fragment`
attempts conservative AST equivalence checks; `unverified` skips them and reports
`semantics_unverified`. It does not select or execute tests and does not change
result/answer comparison.

The optional [dataset replay test](../tests/agents/test_validation_dataset_replay.py)
checks trace generation and splitting with reasoning enabled and disabled, then
recomputes saved-trace grades and exercises student tool/final-answer evaluation.
Supply a prepared dataset directory and previous traces containing actual tool
responses for its queries:

```bash
VALIDATION_REPLAY_DATASET=/path/to/dataset/version \
VALIDATION_REPLAY_TRACES=/path/to/previous/teacher_traces.jsonl \
python -m pytest -q tests/agents/test_validation_dataset_replay.py
```

In replay mode the test requires exact scene/query matches in saved execution
evidence and writes only to pytest's temporary directory. It stubs the tokenizer
and generation runtime; it does not check model quality or actual token lengths.

To check fresh scene imports and real query execution, set `NEO4J_PASSWORD` and
replace the saved-trace input with an explicit live URI:

```bash
VALIDATION_LIVE_NEO4J_URI=bolt://neo4j:7683 \
VALIDATION_REPLAY_DATASET=/path/to/dataset/version \
python -m pytest -q tests/agents/test_validation_dataset_replay.py \
  tests/cypher/test_result_validation_live.py
```

The live tests support Community Edition's existing database. Scene imports and
test graphs run in explicit transactions that are always rolled back.
Each test compares the committed graph's nodes, relationships, and properties
before and after. Controlled repair syntax errors use separate `EXPLAIN`
transactions because parser errors would otherwise abort the scene transaction.
The database defaults to `neo4j`; set `VALIDATION_LIVE_NEO4J_DATABASE` to override it.

New contract-backed training traces require a passing retrieval check, correct
final answer, and grounded final answer. Unresolved or unavailable checks are
excluded rather than labeled proven incorrect. Static-only contract traces are
not accepted as validated training examples. Historical records retain their
legacy behavior until explicitly migrated. Saved contract-backed trace audits
recompute from saved successful tool evidence; missing evidence is not checked.

During teacher generation, tool responses expose only safety/schema or execution
failures and successful database rows. They never expose decoding, grounding, or
expected-result validation statuses, and rows are not withheld because a result
differs from gold. Internal retrieval grading uses the latest successful execution
after the final answer is generated. Earlier exploratory executions do not need
to satisfy the complete retrieval contract, and passing intermediate evidence is
not selected using the expected solution.

The optional teacher optimization `recover_earlier_retrievals` runs after
generation. It can retain an earlier complete retrieval when every subsequent
call is demonstrably diagnostic from public query/schema constraints. It keeps
all messages, selects the latest non-diagnostic task retrieval without consulting
expected values or the final answer, and only then checks correctness and final
grounding. A newer conflicting or unresolved task retrieval blocks recovery.
Unsupported syntax and incomplete feedback cannot establish recovery evidence.
Recovered traces record `retrieval_evidence_policy: earlier_retrievals_v1` and
`grounding_tool_call_index`; saved-trace evaluation and rescoring recompute that
selection and reject inconsistent markers. Unmarked traces retain the latest
successful-execution policy. See [dataset generation](dataset_generation.md) for
the supported diagnostics and reporting.

Student reports expose result status counts and coverage alongside accuracy.
Unresolved/not-checked records stay in total denominators. End-to-end reports
separate final-answer accuracy, retrieval match, and grounded-answer rate. The
workflow HTML detail view includes those statuses and semantic evidence.

## Audit, migration, and rescoring

The decoder supports typed mixed-layer UNION columns even when their alias
matches one role; explicit configured column selectors remain authoritative.
Literal list projections retain tuple positions, indexed collections retain
target lineage, and relationship variables preserve auxiliary aggregate types.
Class equalities joined by OR establish the same coverage as an IN filter.
Known `toLower`/`CONTAINS` searches can prove an empty role from public constraints;
positive LIMITs preserve empty-result evidence, while LIMIT 0 and incomplete
nonempty results cannot establish full pooled-role coverage. Unknown syntax and
ambiguous same-type outputs remain unresolved. QA target origins belong in the
validation contract so source and target columns can be distinguished without
consulting expected answers.

For a completed teacher dataset, use the saved-evidence revalidation command in
[dataset generation](dataset_generation.md#revalidate-saved-teacher-traces-without-generation)
to update finalized generation files without additional teacher requests.

These commands do not call model APIs:

```bash
python scripts/cypher/audit_validation.py audit

python scripts/cypher/audit_validation.py migrate \
  --records historical/test.jsonl --output /tmp/test.validation-v1.jsonl

python scripts/cypher/audit_validation.py rescore \
  --records historical/raw-generation.report.json \
  --source-records /tmp/test.validation-v1.jsonl \
  --output /tmp/rescored.validation-v1.json
```

`audit` validates catalog coverage, source query parsing, formats, goals and
binding declarations. `--templates` selects alternate catalogs for audit/migrate;
their validation files follow the same sibling discovery rules. Migrating a
historical record requires a selected template **and** a trusted saved query that
reproduces its parsed operations after recovering parameters. A matching
`query_type` alone is insufficient. Existing per-record comparison metadata is
retained. Missing grounding, changed queries, inconsistent gold goals or formats
are reported as migration issues; original labels are never rewritten. Such
records must be resolved or regenerated as a new dataset version before rescoring.

`rescore` uses complete saved rows and generated answers. `--source-records`
associates records by exact UID. Its output contains before/after matches, status
counts, contract/source/implementation hashes, changed grades, and reasons for
unavailable checks. Reports with missing/truncated rows need fresh execution;
missing generations need fresh inference. Neither can be repaired by assuming an
empty result. All outputs are exclusively created; existing files are protected.

Resolved contracts, contract hashes, source config provenance, and versions travel
with seeds, traces and rendered splits. Validation defaults, declarations
and implementation fingerprints enter dataset stage provenance,
teacher-resume identities and evaluation cache identities. Relevant changes
invalidate validation reuse. Expected values and private contracts are never
inserted into model messages or repair feedback.
