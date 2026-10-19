# Validation Contracts

Contracts define required query results, answer formats, and PDDL goals.
Use this guide when customizing questions or interpreting validation failures.
For evaluation commands, see [Validation](validation.md).

## Configuration files

| Location under `configs/question_templates/` | Purpose |
| --- | --- |
| Main question catalogs | Wording, variables, queries, and sampling |
| `validation/defaults.yaml` | Shared validation policies |
| `validation/<question-catalog-name>.yaml` | Per-question result, answer, and goal rules |
| `synthetic_reasoning/` | Synthetic retrieval explanations, described in [Reasoning experiments](reasoning.md) |

Validation files use `version: 1` and a `templates` mapping keyed by
`query_type`. Per-template settings override shared defaults. Lists are replaced.
Unknown fields, duplicate keys, missing contracts, and invalid goals are errors.

By default, validation files are discovered beside the question catalog.
Override their locations in a question-set entry when needed:

```yaml
question_sets:
  - name: cypher_qa
    question_templates_path: cypher_qa_3dsg.yaml
    experiment_profile_path: cypher_qa_template.yaml
    validation_config_path: configs/question_templates/validation/cypher_qa_3dsg.yaml
    validation_defaults_path: configs/question_templates/validation/defaults.yaml
```

## Result and answer rules

| Field | Controls |
| --- | --- |
| `query_policy` | Read-only safety and optional refusal literal |
| `result` | Result kind, types, row limits, column mapping, and comparison |
| `answer` | Accepted formats, empty representation, and comparison |
| `goal` | PDDL predicates, arguments, and logical composition |

Result kinds include `scalar`, `set`, `multiset`, `list`, `tuples`, and
`role_sets`. Use tuples when associations across columns matter. Use role sets
for independent collections such as destinations and waypoints.

Answer formats include `str`, `int`, `float`, `point`, `bool`, and nested
`set[...]` or `list[...]` forms. Sets ignore duplicates, lists preserve order,
and multisets preserve counts. Symbols are case-sensitive.

Set `result.comparison` for query-result correctness and `answer.comparison`
for final-answer correctness. Configure both when they need the same tolerance
or ranking policy.

### Count example

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

A count must return one integer, including zero. Missing rows, null, boolean,
and fractional results fail the shape check. Column aliases do not affect a
single unambiguous value.

### Point-set example

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

A point requires three finite coordinates. Supported representations include
coordinate triples, Neo4j points, `{x, y, z}` objects, and `POINT(x y z)` text.

## PDDL goals

Contracts declare named result roles and the goal built from them:

```yaml
pddl_visit_object_symbol:
  result:
    kind: role_sets
    roles:
      object:
        type: node_symbol
        layers: [Object]
        property: nodeSymbol
        oracle_source: value
        origin: {nodeSymbol: {binding: object}}
        cardinality: set
  goal:
    fact: visited-object
    arguments: [{result: object}]
    multiple: or
  answer:
    empty: ''
    comparison: {mode: pddl_equivalence}
```

Bindings refer to sampled question variables. Origins describe query inputs.
`oracle_source` selects the reference field used during seed preparation.

Use `multiple: or` for a choice among matching entities and `multiple: and`
when all are required. Compound goals use `operator: and|or|not` with `clauses`.
`negate_each` negates individual facts. Predicate and role types must agree
with the robot domain.

See the configured
[PDDL catalog](../configs/question_templates/validation/cypher_pddl_3dsg.yaml)
for goals with multiple roles and nested logic.

## Column mapping and ambiguity

Default mapping uses query provenance and unique compatible types. Renaming an
alias does not change a result's meaning. Explicit role names remain
significant. Ambiguous same-type columns are reported as `mapping_unresolved`.
Expected answers are never used to choose a mapping.

For a controlled output layout, set `mapping: explicit` and provide each role's
`column` or `position`. Nested map fields can use `path`:

```yaml
result:
  kind: role_sets
  mapping: explicit
  accepted_forms: [maps, scalar_columns]
  roles:
    object: {type: node_symbol, layers: [Object], column: payload, path: [target]}
    room: {type: node_symbol, layers: [Room], column: payload, path: [region]}
```

Extra diagnostic columns are allowed when their types can be inferred.
Unknown expressions require a typed `auxiliary` declaration. Invalid values
remain errors. They are not silently discarded.

## Read validation outcomes

| Outcome | Meaning |
| --- | --- |
| `pass` | Returned values satisfy the contract and reference result |
| `mismatch` | Returned values differ from the reference |
| `invalid_result_shape` | Rows or values violate the required shape or types |
| `mapping_unresolved` | Columns cannot be assigned to roles unambiguously |
| `invalid_expected_solution` | The reference result or goal is invalid |
| `not_checked` | Required validation evidence is unavailable |

Final answers need exactly one complete `<answer>` block, valid formatting,
correct values, and grounding in the actual tool result. Unresolved and
unchecked examples cannot become validated training traces. Evaluation reports
retain them in total denominators and show coverage separately.

## Apply contract changes

Prepared datasets retain their saved contracts. Prepare a new dataset version
to adopt changed defaults. Reusing prepared inputs for another teacher keeps
the saved contracts.

Use [Dataset audits](dataset_audits.md) to check catalogs or regrade compatible
saved evidence. Rebuild splits after applying dataset revalidation.
