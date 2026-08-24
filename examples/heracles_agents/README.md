# Heracles Agents Example Integration

These files are project-local examples for running distilled 3DSG agent models
with the external `heracles_agents` package.

- `ollama_distilled_cypher_qa_config.yaml` uses Ollama-native tool calls for direct Cypher QA.
- `ollama_distilled_cypher_pddl_config.yaml` is the future PDDL profile with Ollama-native tool calls.
- `model_lists/distilled_students.yaml` lists expected local Ollama model names.

Use the native Ollama config for deployment when the student model supports tool
calls. Use the custom fallback for older models or provider paths that do not
surface native tool calls.

Set `CYPHER_3DSG_AGENT_PATH` to this repository root before using the
Heracles-aligned configs, so `heracles_agents` can load the distilled prompt
YAML files.
