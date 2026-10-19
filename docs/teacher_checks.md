# Teacher Checks and Pilots

Check an OpenRouter teacher before generating a complete dataset. Commands use
the [Docker stack](../docker/README.md). Live checks require
`OPENROUTER_API_KEY`, prepared seeds, and Neo4j.

## Choose a check

| Command | What it measures | Remote generation |
| --- | --- | --- |
| `preflight` | Catalog availability, tool/reasoning settings, and listed prices | None |
| `check` | One conversation's correctness, reasoning availability, cost, and time | One seed |
| `generate --pilot` | Acceptance, reasoning coverage, and estimated cost across tasks and scenes | Sampled seeds |

Catalog support does not guarantee readable reasoning in every response.
Compare the saved results using the same seeds and settings.

## Check catalog support

```bash
docker/stack.sh main python scripts/agents/generate_traces.py preflight \
  --teacher-config configs/teachers/teacher_deepseek_v4_1_flash.yaml
```

This does not generate tokens or require Neo4j.

## Check one conversation

```bash
docker/stack.sh main python scripts/agents/generate_traces.py check \
  --teacher-config configs/teachers/teacher_deepseek_v4_1_flash.yaml \
  --data-config configs/data.yaml --dataset-version SOURCE_VERSION \
  --max-tokens 2048 --max-requests 2 --max-cost-usd 0.10
```

The command selects one answerable QA seed by default. It replaces Neo4j's
active scene, runs the conversation, and writes a report under
`outputs/teacher_checks/`. It does not create training traces.

| Option | Purpose |
| --- | --- |
| `--seed-uid` | Select a specific QA or PDDL seed |
| `--max-requests` | Bound model requests, including repair turns |
| `--max-tokens` | Bound each response |
| `--max-cost-usd` | Set the spending limit, default `$0.10` for `check` |
| `--no-include-reasoning` | Check without retained reasoning |
| `--reasoning-effort default` | Use the provider default instead of configured effort |
| `--output`, `--force` | Choose a report path and allow replacement |

Checks request reasoning by default. Reports distinguish readable text,
summaries, encrypted blocks, and missing reasoning at tool and final-answer
stages. Encrypted blocks are unsuitable as training text.

Inspect the generated queries, answers, and reasoning alongside validation
results. A token-truncated response is inconclusive. Increase the token limit
when the report shows `finish_reason: length`.

One conversation gives only a rough cost and time estimate. Use a pilot to
measure coverage and rejection rates across question types.

## Run a coverage pilot

```bash
docker/stack.sh main python scripts/agents/generate_traces.py generate \
  --data-config configs/data.yaml --dataset-version SOURCE_VERSION \
  --teacher-config configs/teachers/teacher_deepseek_v4_1_flash.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --pilot --pilot-per-task-type 2 --pilot-extra-per-tag 2 \
  --sample-seed 42 --max-cost-usd 3 --preflight
```

Run the same command with each candidate's teacher profile. Retain the printed
dataset version for each pilot.

Sampling covers every task type and its represented scenes. Scene coverage may
increase the requested per-type count. Extra tag samples cover cases such as
invalid inputs, negation, and ordering. Compare `seed_manifest.yaml` to confirm
that candidates used the same inputs.

For a smaller first pilot, use `--pilot-per-task-type 1` and
`--pilot-extra-per-tag 0`. Scene coverage still applies.

`pilot_report.yaml` summarizes acceptance, reasoning coverage, training-length
usability, rejection reasons, billed cost, and full-dataset cost estimates.
Missing costs or unsampled groups make estimates incomplete.

## Spending limits

Spending limits estimate the next request from catalog prices before sending
it. Actual provider charges can exceed those estimates. Missing prices or
unknown billed costs stop further budgeted requests.

Costs include rejected and unfinished conversations. Reasoning tokens are
already included in completion billing. Compare billed totals, acceptance, and
usable reasoning together when selecting a teacher.

## Resume a stopped run

Resume against the dataset version printed by the stopped run:

```bash
docker/stack.sh main python scripts/agents/generate_traces.py generate \
  --data-config configs/data.yaml --dataset-version PILOT_VERSION \
  --teacher-config configs/teachers/teacher_deepseek_v4_1_flash.yaml \
  --student-config configs/students/student_qwen3.5_4b.yaml \
  --pilot --pilot-per-task-type 2 --pilot-extra-per-tag 2 \
  --sample-seed 42 --max-cost-usd 3 --resume
```

Keep data, teacher, tokenizer, prompts, schema, and sampling settings unchanged.
The spending limit can increase. Saved responses are reused while database
queries are rerun.

A `pending_request.json` file blocks retry when a request's outcome was not
saved. Reconcile it with OpenRouter before retrying to avoid duplicate charges.
The same resume rules apply to full remote generation.

A stopped pilot is not a complete training dataset. After choosing a teacher,
run [full generation](teacher_generation.md#generate-traces) without `--pilot`.
