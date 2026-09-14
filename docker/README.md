# Docker Development Stack

Docker is the recommended interface for dataset generation, training,
validation, and GGUF export.

## Services

| Service | Role | Started by `setup` / `up` |
| --- | --- | --- |
| `main` | Project scripts, tests, LoRA/QLoRA training, evaluation, and export | Yes |
| `neo4j` | 3D Scene Graph database used for tool execution and validation | Yes |
| `ollama` | Optional inference service for project-split GGUF validation | No |

The Ollama service is behind a Compose profile. Heracles Benchmark uses a
separate stack, model cache, scene, questions, and metrics pipeline under
`examples/herables_benchmark/`.

## Requirements

- NVIDIA GPU and compatible driver
- Docker Engine
- Docker Compose v2
- NVIDIA Container Toolkit

Check the host and container GPU setup at any time:

```bash
docker/stack.sh check
```

## First run

Create the local environment file. Add `OPENROUTER_API_KEY` only if the remote
teacher will be used:

```bash
cp docker/.env.example docker/.env
```

Validate prerequisites, build the main image, and start `main` plus Neo4j:

```bash
docker/stack.sh setup
```

Load the default scene and inspect service status:

```bash
docker/stack.sh load-neo4j
docker/stack.sh status
```

## Automated workflows

### Train and validate

```bash
docker/stack.sh demo-train
```

This workflow performs scene loading, dataset and trace preparation, grouped
splitting, base-model validation, adapter training, fine-tuned validation, and
HTML report generation. For its stages and overrides, see
[Student training](../docs/training.md#automated-training-workflow).

### Export and deploy

After training has produced an adapter:

```bash
docker/stack.sh demo-deploy
```

This workflow exports and registers BF16, Q8_0, and Q4_K_M variants, validates
them on the challenge split, and regenerates the report. See
[Ollama GGUF export](../docs/ollama_export.md#register-models-in-the-project-stack).

## Daily use

```bash
docker/stack.sh up
docker/stack.sh main python scripts/evaluate_traces.py \
  --data-config configs/data.yaml --split train
docker/stack.sh shell
docker/stack.sh logs main
docker/stack.sh logs neo4j
docker/stack.sh down
```

The project virtual environment is activated automatically for `main` and
`shell`, so use `python` directly inside the container.

Start Ollama only when validating exported GGUFs on project splits:

```bash
docker/stack.sh ollama-up
docker/stack.sh ollama list
```

## Command reference

### Setup and lifecycle

| Command | Action |
| --- | --- |
| `setup` | Check prerequisites, build `main`, and start `main` and `neo4j` |
| `check` | Check Docker, Compose, NVIDIA GPU, and container GPU access |
| `build` | Rebuild the main image |
| `up` | Start `main` and `neo4j` |
| `ollama-up` | Start the optional project Ollama service |
| `down` | Stop containers without deleting persistent volumes |
| `restart [SERVICE...]` | Restart all or selected services |

### Project execution

| Command | Action |
| --- | --- |
| `main COMMAND [ARGS...]` | Run a command in the `main` container |
| `shell` | Open an interactive Bash shell in `main` |
| `exec COMMAND [ARGS...]` | Compatibility alias for `main` |
| `ollama COMMAND [ARGS...]` | Run the CLI in the optional Ollama container |
| `load-neo4j [SCENE]` | Load a scene into the project Neo4j service |
| `demo-train` | Run the automated training and validation workflow |
| `demo-deploy` | Export, register, and validate deployable GGUF variants |

### Inspection

| Command | Action |
| --- | --- |
| `status` / `ps` | Show service status |
| `logs [SERVICE...]` | Follow all or selected service logs |
| `config` | Validate and print the resolved Compose configuration |

Run `docker/stack.sh help` for the terminal version of this reference and the
supported `DEMO_*` and `DEPLOY_*` environment variables.

## Network addresses

| Service | From `main` | From the host |
| --- | --- | --- |
| Neo4j Bolt | `neo4j://neo4j:7683` | `neo4j://127.0.0.1:7683` |
| Neo4j HTTP | — | `http://127.0.0.1:7469` |
| Ollama | `http://ollama:11434` | Not published |

The default Neo4j username is `neo4j` and the password is `neo4j_pw`.

## Image and persistent data

The main image provides CUDA-enabled PyTorch, installs project dependencies in
`/home/user/workspace/.venv`, and builds llama.cpp under
`/home/user/workspace/llama.cpp`.

The repository is bind-mounted into `main`, so generated outputs remain on the
host. Compose preserves these named volumes across `down`:

- `3dsg-cypher-agent-huggingface-cache`
- `3dsg-cypher-agent-neo4j-data`
- `3dsg-cypher-agent-neo4j-logs`
- `3dsg-cypher-agent-ollama-cache` when Ollama is used

## Direct Compose inspection

The wrapper is the supported interface. Inspect the resolved Compose file with:

```bash
docker compose -f docker/docker-compose.yaml config --quiet
```

## Related guides

- [Dataset generation](../docs/dataset_generation.md)
- [Student training](../docs/training.md)
- [Validation](../docs/validation.md)
- [Ollama GGUF export](../docs/ollama_export.md)
- [HTML workflow report](../docs/workflow_report.md)
- [Heracles Benchmark integration](../examples/herables_benchmark/README.md)
