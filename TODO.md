# Cypher 3DSG Agent Distillation Roadmap

## Done

- [x] Add Heracles as `external/heracles` git submodule.
- [x] Extract uHumans2 scene values through `spark_dsg`.
- [x] Generate validated 3DSG Cypher seed data from real scene values.
- [x] Add Heracles Docker Neo4j validation with expected-answer comparison.
- [x] Add canonical agent trace records.
- [x] Add Ollama-native and custom XML render views.
- [x] Add OpenRouter teacher client with `openai/gpt-oss-120b` default.
- [x] Migrate OpenRouter teacher transport to the official Python SDK.
- [x] Replace synthetic teacher trace prompts with real tool-using conversations.
- [x] Add bounded teacher repair attempts with `max_tool_calls`.
- [x] Add optional student-trace pruning for failed repair turns.
- [x] Add grouped split leakage checks.
- [x] Add project-local Heracles Agents example configs.

## Agent Data

- [ ] Generate accepted teacher traces from `openai/gpt-oss-120b` through OpenRouter.
- [ ] Execute and filter teacher traces against Heracles Neo4j.
- [ ] Save rejected teacher traces with stable reason codes.
- [ ] Cap examples per task family after distillation.
- [ ] Keep refusal/write-safety examples at 5-10% of training data.
- [ ] Tune `max_tool_calls` after reviewing rejected repair attempts.
- [ ] Add a held-out challenge split for selected task families.
- [ ] Add more scenes once available, then add scene-held-out evaluation.

## Training

- [ ] Train the primary Qwen3.5 4B student on Ollama-native traces.
- [ ] Train a Granite 3B student for comparison.
- [ ] Record training seed, dataset hash, adapter path, GPU, runtime, and peak memory.
- [ ] Add early stopping based on grouped eval agent success.

## Evaluation

- [ ] Run agent metrics on grouped test and challenge splits.
- [ ] Report native tool-call parse success, correct tool name, valid Cypher argument,
      read-only safety, execution success, final answer match, tool-call count,
      hallucinated-tool rate, and refusal accuracy.
- [ ] Add tag-level and task-family summaries to agent reports.
- [ ] Add numeric tolerance for point/distance-heavy answers.
- [ ] Add compact failures-only report output.
- [ ] Add guardrails for overly broad generated queries.

## Deployment

- [ ] Merge or package the selected LoRA adapter for Ollama export.
- [ ] Convert the merged model to GGUF.
- [ ] Quantize Q4_K_M first and benchmark Q5_K_M only if memory allows.
- [ ] Validate the exported model through the project-local Heracles Agents examples.
- [ ] Benchmark on Jetson Orin NX 16GB: model size, peak memory, tokens/sec, median
      latency, p95 latency, and success under `max_iterations: 5`.


challenge.student.raw-generation.report.json
        "cypher_solution": "O0",
        "expected_solution": "<O0>\n",
        "cypher_solution_format_valid": true,
        "cypher_solution_match": false,