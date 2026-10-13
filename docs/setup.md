# Native Setup

This guide prepares a Linux host to run the project without the development
Docker stack. It is the source of truth for native system packages, Python,
PyTorch, Neo4j, llama.cpp, Ollama, and environment variables.

## What to install

Install only the components required by the workflows you intend to run:

| Component | Required for |
| --- | --- |
| Python environment and project package | All project scripts |
| Heracles submodules | Scene loading and SLDP/PDDL validation |
| PyTorch with GPU support | LoRA/QLoRA training, Transformers evaluation, and fine-tuned weight merging |
| Neo4j | Teacher trace execution and raw-generation validation |
| llama.cpp | Hugging Face-to-GGUF conversion and GGUF quantization |
| Ollama | Optional inference and raw-generation validation of exported GGUFs |
| Development dependencies | Tests, Ruff, and contributor checks |

The package requires Python 3.10 or newer. An NVIDIA GPU is expected for the
configured training workflows. Data preparation, static checks, and report
generation can run without one.

## 1. Install system prerequisites

On Ubuntu or Debian, install the tools needed by the Python environment and
llama.cpp build:

```bash
sudo apt-get update
sudo apt-get install --yes \
  build-essential \
  ca-certificates \
  cmake \
  curl \
  git \
  ninja-build \
  python3 \
  python3-pip \
  python3-venv
```

Use equivalent packages on another Linux distribution.

## 2. Clone the repository and submodules

For a new checkout:

```bash
git clone --recursive https://github.com/JorgeDFR/3dsg-cypher-agent.git
cd 3dsg-cypher-agent
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

The recursive update is required because the pinned Heracles Benchmark
submodule contains the Heracles and Heracles Agents revisions used by the main
project.

If a nested submodule uses an SSH URL and the host has no GitHub SSH access,
initialize it through HTTPS without changing the recorded URL:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --recursive
```

Detached HEADs inside submodules are expected. They preserve the revisions
recorded by their parent repositories.

## 3. Create the Python environment

Create one virtual environment at the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Keep `.venv` activated when running commands in the remaining sections.

## 4. Install PyTorch

PyTorch is intentionally not pinned in `pyproject.toml` because the correct
wheel depends on the host GPU, driver, CUDA/ROCm platform, and architecture.
Install it into the project virtual environment after selecting the appropriate
command from the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/).

For CPU-only development tasks, a standard installation is sufficient:

```bash
python -m pip install torch
```

For NVIDIA or AMD training, use the selector's CUDA or ROCm command instead.
On Jetson, use the wheel matched to the installed JetPack release in NVIDIA's
[PyTorch for Jetson guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).

Verify the installed build and GPU visibility:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda)"
```

`CUDA available: True` is required for the repository's normal LoRA/QLoRA
training profiles. Also confirm that the host driver sees the GPU:

```bash
nvidia-smi
```

QLoRA additionally depends on a bitsandbytes build that supports the host
platform. If that is unavailable, use a student profile with
`model.load_in_4bit: false` for ordinary LoRA.

## 5. Install the project package

After installing the correct PyTorch build, install the project and every
optional dependency group used by the documented native workflows:

```bash
python -m pip install -e '.[dev,train,export]'
```

For a smaller installation, choose only the required extras:

| Command | Includes |
| --- | --- |
| `python -m pip install -e .` | Core data generation and validation dependencies |
| `python -m pip install -e '.[train]'` | Accelerate, bitsandbytes, PEFT, and TRL |
| `python -m pip install -e '.[export]'` | Fine-tuned weight merging and export dependencies |
| `python -m pip install -e '.[dev]'` | Pytest and Ruff |

## 6. Configure OpenRouter credentials

The synthetic teacher requires no remote credentials. Export an API key only
when using an OpenRouter teacher:

```bash
export OPENROUTER_API_KEY=your-key
```

Add the variable to a private shell environment file if it must persist across
sessions. Do not add credentials to tracked configuration files.

Verify the configured teacher's availability before generating traces:

```bash
python scripts/agents/generate_traces.py preflight \
  --teacher-config configs/teachers/teacher_deepseek_v4_1_flash.yaml
```

The preflight checks catalog declarations without generating tokens. To verify
readable reasoning and answer correctness, run `generate_traces.py check` with one
dataset conversation and live Neo4j. See
[teacher capability checks](reasoning_traces.md#cheap-teacher-capability-checks)
for request/budget limits, formatted metrics, and generated-answer inspection.

## 7. Install and configure Neo4j

Install Neo4j Community Edition with the official
[native Linux instructions](https://neo4j.com/docs/operations-manual/current/installation/linux/).
The development stack currently uses Neo4j `5.25.1`. Use that version or a
compatible Neo4j 5 release for the closest native equivalent.

The commands below assume the archive layout. Set the extracted installation
path:

```bash
export NEO4J_HOME=/absolute/path/to/neo4j
```

Add these values to `$NEO4J_HOME/conf/neo4j.conf` so the existing project
defaults work unchanged:

```properties
server.bolt.listen_address=:7683
server.bolt.advertised_address=:7683
server.http.listen_address=:7469
server.http.advertised_address=:7469
```

Package-managed installations may place `neo4j.conf` under `/etc/neo4j` and
provide `neo4j` and `neo4j-admin` on `PATH`. Use those locations while keeping
the same settings. Consult Neo4j's
[file-location reference](https://neo4j.com/docs/operations-manual/current/configuration/file-locations/)
for the selected installation method.

Before the first database start, set the development password expected by the
default project commands:

```bash
"$NEO4J_HOME/bin/neo4j-admin" dbms set-initial-password neo4j_pw
```

Start Neo4j in the foreground:

```bash
"$NEO4J_HOME/bin/neo4j" console
```

For an installation managed as a system service, use the service commands from
the Neo4j installation guide instead. In either case, the native connection is:

| Setting | Value |
| --- | --- |
| Bolt URI | `neo4j://127.0.0.1:7683` |
| HTTP URI | `http://127.0.0.1:7469` |
| Username | `neo4j` |
| Password | `neo4j_pw` |
| Database | `neo4j` |

Export the connection values for scripts that read them from the environment:

```bash
export NEO4J_URI=neo4j://127.0.0.1:7683
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=neo4j_pw
export NEO4J_DATABASE=neo4j
export HERACLES_NEO4J_URI="$NEO4J_URI"
export HERACLES_NEO4J_USERNAME="$NEO4J_USERNAME"
export HERACLES_NEO4J_PASSWORD="$NEO4J_PASSWORD"
export ADT4_HERACLES_IP=127.0.0.1
export ADT4_HERACLES_PORT=7683
```

Load the default scene to verify the connection:

```bash
python scripts/graph/load_3dsg_neo4j.py \
  --scene data/scenes/uhumans2_office.json \
  --uri "$NEO4J_URI" \
  --username "$NEO4J_USERNAME" \
  --password "$NEO4J_PASSWORD" \
  --database "$NEO4J_DATABASE"
```

## 8. Install llama.cpp for GGUF export

llama.cpp is not a submodule of this repository. Clone it beside the project so
its frequently changing files do not appear in this Git worktree:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ../llama.cpp
python3 -m venv ../llama.cpp/.venv
../llama.cpp/.venv/bin/python -m pip install --upgrade pip
../llama.cpp/.venv/bin/python -m pip install -r ../llama.cpp/requirements.txt
cmake -S ../llama.cpp -B ../llama.cpp/build -G Ninja
cmake --build ../llama.cpp/build --target llama-quantize
```

Set the absolute checkout path for the project exporter:

```bash
export LLAMA_CPP_DIR=/absolute/path/to/llama.cpp
```

The exporter automatically prefers
`$LLAMA_CPP_DIR/.venv/bin/python` for `convert_hf_to_gguf.py` and searches the
standard CMake output locations for `llama-quantize`. You can override those
paths with `--python-bin`, `--convert-script`, or `--quantize-bin`.

For reproducible exports, pin the llama.cpp checkout to a tested commit instead
of leaving it on the moving default branch.

## 9. Install Ollama for native GGUF validation

Ollama is optional. Install it only when exported GGUF models will be registered
and evaluated on the host. Follow the official
[Ollama Linux guide](https://docs.ollama.com/linux), or use its installer:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the service using the method configured by the installer, then verify the
client and API:

```bash
ollama -v
curl -fsS http://127.0.0.1:11434/api/version
```

Tell project evaluation commands to use the host service:

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
```

Model export and registration remain documented in
[Ollama GGUF export](ollama_export.md). Student validation is documented in
[Validation](validation.md).

## Run project commands natively

With `.venv` active, invoke project entry points directly with
`python scripts/<script>.py`. When a workflow guide shows a container-prefixed
Python command, retain the script and its arguments while omitting the
container wrapper.

Automated stack orchestrators and service lifecycle helpers do not have a
single native equivalent. Run their individual Python stages using the
detailed workflow guides, and manage Neo4j and Ollama with their native service
commands.

## Verify the native environment

Check the package, command entry points, and development dependencies:

```bash
python -c "import cypher_3dsg_agent_distillation"
python scripts/training/train_agent_lora.py --help
python scripts/training/export_agent_ollama.py --help
pytest
ruff check .
```

Verify the optional export toolchain when installed:

```bash
test -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
test -x "$LLAMA_CPP_DIR/build/bin/llama-quantize"
```

See [Configuration and versioning](../README.md#configuration-and-versioning)
for the shared YAML configuration map. Each workflow guide describes the
settings and commands specific to that stage.
