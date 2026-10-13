# Command-line scripts

Scripts are grouped like `src/cypher_3dsg_agent_distillation`:

| Directory | Commands |
| --- | --- |
| `agents/` | Prepare datasets, generate teacher traces, evaluate students and traces |
| `core/` | List artifact versions and migrate versioned artifacts |
| `cypher/` | Generate question templates, analyze repair templates, validate Cypher |
| `graph/` | Extract scene values, edit room labels, repair object-to-place edges, load scenes into Neo4j |
| `reporting/` | Generate workflow reports and summarize validation |
| `training/` | Train, inspect datasets, export to Ollama, and read deployment settings |

Run scripts from the project root, for example:

```bash
python scripts/training/export_agent_ollama.py --help
python scripts/reporting/generate_workflow_report.py --help
```

The shared `_bootstrap.py` adds the local `src/` directory to the Python import
path and configures console output. `docker/stack.sh` uses these same entry points.
