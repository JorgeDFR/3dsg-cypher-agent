# Docker Development Stack

Docker is the default interface for training and export. The normal development
stack contains two services:

- `main`: project scripts, tests, QLoRA training, evaluation, and GGUF export;
- `neo4j`: the 3D scene graph database.

Ollama is intentionally not part of this stack. Local model registration,
inference, and Heracles sweeps use the isolated stack under
`examples/herables_benchmark`, where the Ollama container can be
measured consistently.

## First run

The development stack requires an NVIDIA GPU, a compatible NVIDIA driver,
Docker Engine, Docker Compose v2, and NVIDIA Container Toolkit.

Create the optional local environment file and add credentials such as
`OPENROUTER_API_KEY`:

```bash
cp docker/.env.example docker/.env
```

Validate the host, build the main image, and start `main` plus Neo4j:

```bash
docker/stack.sh setup
```

Load the configured scene and inspect status:

```bash
docker/stack.sh load-neo4j
docker/stack.sh status
```

Run the complete reproducible workflow:

```bash
docker/stack.sh demo
```

The demo loads the scene, prepares questions, creates validated traces with the
synthetic teacher, makes grouped splits, evaluates the base model, trains the
adapter, evaluates the fine-tuned model, and writes the workflow HTML report.
Its inputs can be overridden with `DEMO_SCENE`, `DEMO_DATA_CONFIG`,
`DEMO_TEACHER_CONFIG`, and `DEMO_STUDENT_CONFIG`.

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

## Command reference

Setup and lifecycle:

- `setup`: validate prerequisites, build `main`, and start `main` and `neo4j`;
- `check`: check Docker, Compose, NVIDIA GPU, and container GPU access;
- `build`: rebuild the main image;
- `up`: start `main` and `neo4j`;
- `down`: stop containers without deleting persistent volumes;
- `restart [SERVICE...]`: restart all or selected services.

Project commands:

- `main COMMAND [ARGS...]`: execute a command in `main`;
- `shell`: open an interactive Bash shell in `main`;
- `exec COMMAND [ARGS...]`: compatibility alias for `main`;
- `demo`: run the full training/evaluation workflow;
- `load-neo4j [SCENE]`: load a scene into Neo4j.

Inspection:

- `status` or `ps`: show service status;
- `logs [SERVICE...]`: follow logs;
- `config`: validate and print the resolved Compose configuration.

Run `docker/stack.sh help` for the same reference in the terminal.

## Service addresses

Inside the development network, project scripts use:

```text
neo4j://neo4j:7683
```

From the host:

```text
neo4j://127.0.0.1:7683
http://127.0.0.1:7469
```

The default Neo4j username is `neo4j` and the password is `neo4j_pw`.

## Main image and persistent data

GPU support is required because the configured training path uses QLoRA,
bitsandbytes, and bf16 CUDA execution. The main image supplies CUDA-enabled
PyTorch, installs project dependencies in `/home/user/workspace/.venv`, and
builds llama.cpp under `/home/user/workspace/llama.cpp`.

The repository is bind-mounted into `main`, so generated `outputs` remain on
the host. Compose retains these named volumes across `down`:

- `3dsg-cypher-agent-huggingface-cache`;
- `3dsg-cypher-agent-neo4j-data`;
- `3dsg-cypher-agent-neo4j-logs`.

## Ollama and Heracles benchmark

After exporting GGUF artifacts with the main container, use the dedicated
benchmark launcher to create them in Ollama and run Cypher and/or PDDL sweeps:

```bash
examples/herables_benchmark/run_benchmark.sh --benchmarks cypher
```

See [Ollama GGUF export](ollama_export.md) for artifact creation and the
[benchmark guide](../examples/herables_benchmark/README.md) for model
selection, GPU metrics, reports, and stack utility commands.

## Direct Compose

The helper is the supported interface, but the configuration can be inspected
directly:

```bash
docker compose -f docker/docker-compose.yaml config --quiet
```
