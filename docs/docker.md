# Docker Development Stack

Docker is the default interface for this repository. The Compose stack provides
three services:

- `main`: project scripts, tests, QLoRA training, evaluation, and GGUF export.
- `neo4j`: the 3D scene graph database.
- `ollama`: model registration and local inference.

All three services are managed through `docker/stack.sh`. Run
`docker/stack.sh help` at any time to see the ordered setup guide and command
groups.

## First Run

The stack requires an NVIDIA GPU, a compatible NVIDIA driver, Docker Engine,
Docker Compose v2, and NVIDIA Container Toolkit.

Create the optional local environment file and add credentials such as
`OPENROUTER_API_KEY`:

```bash
cp docker/.env.example docker/.env
```

Validate the host, build the main image, and start the complete stack:

```bash
docker/stack.sh setup
```

`setup` performs the equivalent ordered sequence:

```bash
docker/stack.sh check
docker/stack.sh build
docker/stack.sh up
```

Load the scene configured by the repository and confirm that all services are
running:

```bash
docker/stack.sh load-neo4j
docker/stack.sh status
```

Run the complete reproducible demo after setup:

```bash
docker/stack.sh demo
```

`demo` loads the scene, extracts template values, generates and prepares
questions, creates validated traces with the deterministic synthetic teacher,
creates grouped splits, trains the student adapter, evaluates loss on every
split, validates raw generation on the test and challenge splits, and writes
the HTML workflow report. It does not call a paid teacher API. Override its
inputs with `DEMO_SCENE`, `DEMO_DATA_CONFIG`, `DEMO_TEACHER_CONFIG`, and
`DEMO_STUDENT_CONFIG`.

When using another scene, pass it explicitly:

```bash
docker/stack.sh load-neo4j data/scenes/3dsg_uhumans2.json
```

## Daily Use

Start an existing stack after a reboot or `down`:

```bash
docker/stack.sh up
```

Run project commands in the `main` container:

```bash
docker/stack.sh main python scripts/evaluate_traces.py \
  --data-config configs/data.yaml \
  --split train
```

Open an interactive main-container shell:

```bash
docker/stack.sh shell
```

The project venv is activated automatically for both `main` and `shell`, so use
`python` directly inside the container.

Inspect services or logs:

```bash
docker/stack.sh status
docker/stack.sh logs
docker/stack.sh logs main
docker/stack.sh logs neo4j
docker/stack.sh logs ollama
```

Stop containers while retaining model, database, and cache volumes:

```bash
docker/stack.sh down
```

## Command Reference

### Setup And Lifecycle

- `setup`: validate prerequisites, build `main`, and start all services.
- `check`: run Docker, Compose, NVIDIA GPU, and container-toolkit checks.
- `build`: rebuild the `main` image after dependency or image changes.
- `up`: start `main`, `neo4j`, and `ollama`.
- `down`: stop the stack without deleting persistent volumes.
- `restart [SERVICE...]`: restart all services or selected services.

`setup`, `build`, and `up` always run the complete GPU preflight. Commands that
operate on an existing stack perform lightweight Docker and Compose checks and
report when a required service is not running.

### Main Container

- `main COMMAND [ARGS...]`: execute a project command in `main`.
- `shell`: open an interactive Bash shell in `main`.
- `exec COMMAND [ARGS...]`: backward-compatible alias for `main`.

### Automated Workflow

- `demo`: run the full workflow from the Neo4j scene load through student
  training, evaluation, validation, and HTML report generation, with numbered
  progress sections and normalized script summaries.

Expected Python warnings are grouped into compact Rich tables, and uncaught
script exceptions use Rich tracebacks. Hugging Face, Transformers, Datasets,
TRL, and PEFT progress rendering is disabled for model workflows; the project
shows its own Rich phase, training-step, evaluation, and generation bars
instead. The unauthenticated Hugging Face download warning is suppressed because
the demo uses public model artifacts; other library warnings remain visible.

Examples:

```bash
docker/stack.sh main python scripts/train_agent_lora.py \
  --data-config configs/data.yaml \
  --student-config configs/student_qwen3.5_4b.yaml
```

### Neo4j And Ollama

- `load-neo4j [SCENE]`: load a scene through the main project container.
- `pull-ollama MODEL`: pull a model into the persistent Ollama cache.
- `ollama COMMAND [ARGS...]`: run Ollama CLI from `/models` in its container.

Examples:

```bash
docker/stack.sh pull-ollama qwen3.5:4b
docker/stack.sh ollama list
docker/stack.sh ollama create qwen3.5-4b-3dsg-cypher-agent-lora \
  -f qwen3.5-4b-3dsg-cypher-agent-lora/Modelfile
docker/stack.sh ollama run qwen3.5-4b-3dsg-cypher-agent-lora \
  "List the objects in room R0."
```

See [ollama_export.md](ollama_export.md) for the complete merge, conversion,
quantization, registration, and inference workflow.

### Inspection

- `status`: show Compose service status.
- `logs [SERVICE...]`: follow all logs or selected service logs.
- `config`: validate and print the resolved Compose configuration.
- `ps`: backward-compatible alias for `status`.

## Project Workflow

After `setup` and `load-neo4j`, the normal workflow is:

1. Prepare or generate seed data.
2. Check the teacher model and generate validated traces.
3. Create grouped student splits and evaluate the traces.
4. Train and evaluate the QLoRA student.
5. Export the adapter to GGUF and register it with Ollama.

The README contains the concrete command for every step. All project Python
commands use this form:

```bash
docker/stack.sh main python scripts/<script>.py [ARGS...]
```

## Service Addresses

Inside the Docker network, project scripts use:

```text
neo4j://neo4j:7683
http://ollama:11434
```

From the host, services are available at:

```text
neo4j://127.0.0.1:7683
http://127.0.0.1:7469
http://127.0.0.1:11434
```

The default Neo4j username is `neo4j` and the password is `neo4j_pw`.

## Main Image

GPU support is required. The configured training path uses QLoRA
(`load_in_4bit: true`), bitsandbytes, and bf16 CUDA execution. The main image
uses:

```dockerfile
FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime
```

This image targets modern NVIDIA drivers with CUDA 13 support and
RTX 50-series/Blackwell GPUs. The `train` dependency group requires
`bitsandbytes>=0.49.0` for CUDA 13 and `sm120` support.

Project dependencies are installed in `/home/user/workspace/.venv`. The venv
uses `--system-site-packages` to access the image's CUDA-enabled PyTorch without
modifying the PEP 668 managed system environment or installing another PyTorch
build.

llama.cpp is built under `/home/user/workspace/llama.cpp`; its Python converter
dependencies use the separate `/home/user/workspace/llama.cpp/.venv`.

Host UID and GID values are passed into the image so artifacts written by
`main` retain host-user ownership.

## Persistent Data

The main service bind-mounts host `outputs` at
`/home/user/workspace/3dsg-cypher-agent/outputs`. Ollama sees
`outputs/ollama` at `/models`.

The repository tracks `outputs/ollama/.gitkeep` so the Ollama bind source exists
before Compose starts. Generated files under `outputs` remain ignored by Git.

The stack also creates these named volumes:

- `3dsg-cypher-agent-huggingface-cache`
- `3dsg-cypher-agent-neo4j-data`
- `3dsg-cypher-agent-neo4j-logs`
- `3dsg-cypher-agent-ollama-cache`

`docker/stack.sh down` retains these volumes.

## Direct Compose

The helper script is the supported interface, but the resolved Compose config
can be inspected directly:

```bash
docker compose -f docker/docker-compose.yaml config --quiet
```

To use `docker/.env` explicitly:

```bash
docker compose --env-file docker/.env \
  -f docker/docker-compose.yaml up -d
```
