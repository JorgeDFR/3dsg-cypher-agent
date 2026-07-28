# Heracles Agents Example Integration

These files are project-local examples for running distilled 3DSG agent models
with the external `heracles_agents` package. They intentionally do not modify
`/home/coder/workspace/heracles_devel/heracles_agents`.

- `ollama_distilled_agent_config.yaml` uses Ollama-native tool calls.
- `ollama_custom_fallback_agent_config.yaml` uses the custom XML-style fallback.
- `model_lists/distilled_students.yaml` lists expected local Ollama model names.

Use the native Ollama config for deployment when the student model supports tool
calls. Use the custom fallback for older models or provider paths that do not
surface native tool calls.

