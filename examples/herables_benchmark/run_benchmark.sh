#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yaml"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
NVIDIA_TEST_IMAGE="${NVIDIA_TEST_IMAGE:-ubuntu:24.04}"
if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
else
  BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-python3}"
fi

HERACLES_BENCHMARK_ROOT="${PROJECT_ROOT}/external/heracles_benchmark"
BENCHMARK_CONFIG_RELATIVE="${BENCHMARK_CONFIG:-configs/benchmark.yaml}"
MODEL_MANIFEST_REQUESTED="${BENCHMARK_MODELS:-examples/herables_benchmark/config/model_lists/students.yaml}"
HERACLES_BENCHMARK_OUTPUT_ROOT="${PROJECT_ROOT}/outputs/heracles_benchmark"
RUNTIME_DIR="${HERACLES_BENCHMARK_OUTPUT_ROOT}/.runtime"
BENCHMARK_CONFIG_HOST="${RUNTIME_DIR}/benchmark.yaml"
BENCHMARK_CONFIG_IN_CONTAINER="/home/benchmark/workspace/benchmark/configs/3dsg-cypher-agent.yaml"
MODEL_MANIFEST_HOST=""
OLLAMA_CONTAINER_NAME="3dsg-heracles-benchmark-ollama"

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-3dsg-heracles-benchmark}"
export BENCHMARK_IMAGE="${BENCHMARK_IMAGE:-${COMPOSE_PROJECT_NAME}-main:latest}"
export HERACLES_BENCHMARK_OUTPUT_ROOT
export BENCHMARK_CONFIG_HOST MODEL_MANIFEST_HOST

KEEP_STACK=0
SERVICES_STARTED=0
CLEANED_UP=0
COMPOSE=()

log() { printf '\n==> %s\n' "$*"; }
info() { printf 'INFO: %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: examples/herables_benchmark/run_benchmark.sh [COMMAND] [OPTIONS]

Commands:
  run                       Run QA and PDDL from the selected benchmark config (default).
  build                     Rebuild the benchmark image.
  models                    Create enabled project models in benchmark Ollama.
  ollama COMMAND [ARGS...]  Run the Ollama CLI in the benchmark container.
  status                    Show benchmark service status.
  down                      Stop the stack; the model cache is retained.
  config                    Validate and print the resolved Compose configuration.

Options:
  --config PATH             Config below the Heracles Benchmark configs/ directory.
                            Default: configs/benchmark.yaml
  --models PATH             Model list in this repository. Default:
                            examples/herables_benchmark/config/model_lists/students.yaml
  --keep-stack              Leave services running after a benchmark run.
  -h, --help                Show this help.

Questions, scene data, benchmark code, and compatible package revisions come
from the pinned external/heracles_benchmark submodule.
EOF
}

resolve_inputs() {
  "${BENCHMARK_PYTHON}" -c 'import yaml' >/dev/null 2>&1 ||
    die "${BENCHMARK_PYTHON} cannot import PyYAML; install this project before running."
  [[ -d "${HERACLES_BENCHMARK_ROOT}" ]] ||
    die "Heracles Benchmark submodule is missing. Run git submodule update --init --recursive."

  [[ "${BENCHMARK_CONFIG_RELATIVE}" != /* ]] ||
    die "--config must be relative to the Heracles Benchmark checkout."
  local source_config="${HERACLES_BENCHMARK_ROOT}/${BENCHMARK_CONFIG_RELATIVE}"
  [[ -f "${source_config}" ]] || die "Benchmark config not found: ${source_config}"
  source_config="$(cd -- "$(dirname -- "${source_config}")" && pwd)/$(basename -- "${source_config}")"
  case "${source_config}" in
    "${HERACLES_BENCHMARK_ROOT}"/configs/*.yaml | "${HERACLES_BENCHMARK_ROOT}"/configs/*.yml) ;;
    *) die "--config must select a YAML file below the benchmark configs/ directory." ;;
  esac

  if [[ "${MODEL_MANIFEST_REQUESTED}" == /* ]]; then
    MODEL_MANIFEST_HOST="${MODEL_MANIFEST_REQUESTED}"
  else
    MODEL_MANIFEST_HOST="${PROJECT_ROOT}/${MODEL_MANIFEST_REQUESTED}"
  fi
  [[ -f "${MODEL_MANIFEST_HOST}" ]] || die "Model list not found: ${MODEL_MANIFEST_HOST}"
  MODEL_MANIFEST_HOST="$(cd -- "$(dirname -- "${MODEL_MANIFEST_HOST}")" && pwd)/$(basename -- "${MODEL_MANIFEST_HOST}")"
  case "${MODEL_MANIFEST_HOST}" in
    "${PROJECT_ROOT}"/*) ;;
    *) die "--models must select a model list inside this repository." ;;
  esac
  export MODEL_MANIFEST_HOST

  for required in \
    scripts/benchmark_manifest.py \
    scripts/benchmark_console.py \
    docker/benchmark/benchmark_runner.sh \
    external/heracles/heracles/examples/load_scene_graph.py \
    external/heracles_agents/examples/experiment_runner.py; do
    [[ -f "${HERACLES_BENCHMARK_ROOT}/${required}" ]] ||
      die "The selected Heracles Benchmark checkout is incomplete: missing ${required}"
  done

  mkdir -p "${RUNTIME_DIR}" "${HERACLES_BENCHMARK_OUTPUT_ROOT}"
  "${BENCHMARK_PYTHON}" "${SCRIPT_DIR}/model_manifest.py" \
    --models-path "${MODEL_MANIFEST_HOST}" \
    --benchmark-config "${source_config}" \
    --output-config "${BENCHMARK_CONFIG_HOST}" \
    --ollama-container-name "${OLLAMA_CONTAINER_NAME}" \
    --format tsv >/dev/null

  info "Questions and scene: ${BENCHMARK_CONFIG_RELATIVE}"
  info "Models: ${MODEL_MANIFEST_HOST#"${PROJECT_ROOT}/"}"
}

check_docker() {
  command -v docker >/dev/null 2>&1 || die "Docker is not installed or not in PATH."
  docker version >/dev/null 2>&1 || die "The Docker daemon is not available."
  docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required."
}

check_required_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required."
  nvidia-smi -L >/dev/null 2>&1 || die "No usable NVIDIA GPU was detected."
  log "Testing NVIDIA GPU access from Docker."
  local output
  if ! output="$(docker run --rm --gpus all "${NVIDIA_TEST_IMAGE}" nvidia-smi 2>&1)"; then
    [[ -z "${output}" ]] || printf '%s\n' "${output}" | tail -n 8 >&2
    die "Docker cannot expose the NVIDIA GPU. Configure NVIDIA Container Toolkit."
  fi
  info "NVIDIA GPU access is available to the benchmark containers."
}

build_compose_command() {
  COMPOSE=(docker compose --progress "${COMPOSE_PROGRESS:-quiet}")
  [[ ! -f "${ENV_FILE}" ]] || COMPOSE+=(--env-file "${ENV_FILE}")
  COMPOSE+=(-f "${COMPOSE_FILE}")
}

check_container_name() {
  local owner_project
  if ! docker container inspect "${OLLAMA_CONTAINER_NAME}" >/dev/null 2>&1; then
    return
  fi
  owner_project="$(docker inspect -f '{{ index .Config.Labels "com.docker.compose.project" }}' "${OLLAMA_CONTAINER_NAME}" 2>/dev/null || true)"
  [[ "${owner_project}" == "${COMPOSE_PROJECT_NAME}" ]] ||
    die "Container ${OLLAMA_CONTAINER_NAME} belongs to another Compose project."
}

build_image() {
  log "Building the benchmark integration image."
  "${COMPOSE[@]}" build main
}

start_ollama() {
  check_container_name
  "${COMPOSE[@]}" up -d --wait ollama
  SERVICES_STARTED=1
}

start_services() {
  check_container_name
  log "Starting Neo4j, Ollama, and the read-only metrics proxy."
  "${COMPOSE[@]}" up -d --wait neo4j ollama docker-socket-proxy
  SERVICES_STARTED=1
}

provision_models() {
  log "Validating and creating enabled project models."
  local records
  records="$(
    "${COMPOSE[@]}" run --rm --no-deps --entrypoint python main \
      /home/benchmark/workspace/model_manifest.py \
      --models-path /home/benchmark/workspace/integration/models.yaml \
      --artifacts-root /models \
      --format tsv
  )"
  local -a provisioning_records=()
  mapfile -t provisioning_records <<<"${records}"
  ((${#provisioning_records[@]} > 0)) || die "The model list returned no enabled models."

  local record model modelfile
  local provisioned=0
  for record in "${provisioning_records[@]}"; do
    [[ "${record}" == *$'\t'* ]] || die "Invalid model provisioning record: ${record}"
    model="${record%%$'\t'*}"
    modelfile="${record#*$'\t'}"
    info "Creating ${model} from ${modelfile}."
    "${COMPOSE[@]}" exec -T ollama ollama create "${model}" -f "${modelfile}" </dev/null
    "${COMPOSE[@]}" exec -T ollama ollama show "${model}" </dev/null >/dev/null ||
      die "Ollama model verification failed after create: ${model}"
    ((provisioned += 1))
  done
  info "Provisioned and verified ${provisioned} enabled Ollama model(s)."
}

cleanup() {
  if ((CLEANED_UP == 0 && SERVICES_STARTED == 1 && KEEP_STACK == 0)); then
    CLEANED_UP=1
    log "Stopping benchmark services; the Ollama cache is retained."
    "${COMPOSE[@]}" down --remove-orphans || true
  fi
}

on_interrupt() { cleanup; exit 130; }

run_sweep() {
  trap cleanup EXIT
  trap on_interrupt INT TERM
  build_image
  start_services
  provision_models
  log "Running the current Heracles Benchmark QA and PDDL experiments."
  "${COMPOSE[@]}" run --rm --no-deps main \
    /home/benchmark/workspace/benchmark/docker/benchmark/benchmark_runner.sh \
    0 1 "${BENCHMARK_CONFIG_IN_CONTAINER}"
  printf '\nBenchmark output root:\n  %s\n' "${HERACLES_BENCHMARK_OUTPUT_ROOT}"
}

command=run
if (($# > 0)); then
  case "$1" in
    run | build | models | ollama | status | down | config) command="$1"; shift ;;
    -h | --help) usage; exit 0 ;;
  esac
fi

while (($# > 0)); do
  case "$1" in
    --config) (($# >= 2)) || die "--config requires a path."; BENCHMARK_CONFIG_RELATIVE="$2"; shift 2 ;;
    --models) (($# >= 2)) || die "--models requires a path."; MODEL_MANIFEST_REQUESTED="$2"; shift 2 ;;
    --keep-stack) KEEP_STACK=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *)
      if [[ "${command}" == "ollama" ]]; then break; fi
      die "Unknown option: $1"
      ;;
  esac
done

resolve_inputs
check_docker
if [[ "${command}" == "run" || "${command}" == "models" || "${command}" == "ollama" ]]; then
  check_required_gpu
fi
build_compose_command
"${COMPOSE[@]}" config --quiet || die "The benchmark Compose configuration is invalid."

case "${command}" in
  config) "${COMPOSE[@]}" config ;;
  status) "${COMPOSE[@]}" ps ;;
  down) "${COMPOSE[@]}" down --remove-orphans ;;
  build) build_image ;;
  run) run_sweep ;;
  models)
    trap cleanup EXIT
    build_image
    start_ollama
    provision_models
    ;;
  ollama)
    (($# > 0)) || die "Usage: run_benchmark.sh ollama COMMAND [ARGS...]"
    trap cleanup EXIT
    start_ollama
    "${COMPOSE[@]}" exec -T ollama ollama "$@"
    ;;
esac
