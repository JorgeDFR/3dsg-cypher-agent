#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/docker-compose.yaml}"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
NVIDIA_TEST_IMAGE="${NVIDIA_TEST_IMAGE:-ubuntu:24.04}"
OUTPUT_DIR_IN_CONTAINER="output/model_sweep"
OUTPUT_DIR_ON_HOST="${PROJECT_ROOT}/outputs/heracles_benchmark/model_sweep"
MODEL_MANIFEST_IN_CONTAINER="/home/benchmark/workspace/heracles_agents/examples/experiments/ollama/model_lists/students.yaml"
MODEL_ARTIFACTS_IN_CONTAINER="/models"
OLLAMA_CONTAINER_NAME="3dsg-heracles-benchmark-ollama"

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-3dsg-heracles-benchmark}"
export BENCHMARK_IMAGE="${BENCHMARK_IMAGE:-${COMPOSE_PROJECT_NAME}-benchmark-ollama:latest}"

RUN_CYPHER=1
RUN_PDDL=0
KEEP_STACK=0
SERVICES_STARTED=0
CLEANED_UP=0

usage() {
  cat <<'EOF'
Usage: examples/herables_benchmark/run_benchmark.sh [COMMAND] [OPTIONS]

Commands:
  run                       Run the model sweep (default).
  build                     Force-rebuild the Heracles benchmark image.
  models                    Create all enabled local models in benchmark Ollama.
  ollama COMMAND [ARGS...]  Run the Ollama CLI in the benchmark container.
  status                    Show benchmark service status.
  down                      Stop the benchmark stack; model cache is retained.
  config                    Validate and print the Compose configuration.

Run options:
  --benchmarks cypher       Run only Cypher QA (default).
  --benchmarks pddl         Run only Cypher-PDDL.
  --benchmarks both         Run both experiment sets.
  --keep-stack              Leave services running after the benchmark.
  -h, --help                Show this help.

If `run` is started in a terminal without --benchmarks, an interactive
Cypher/PDDL/both menu is shown. A working NVIDIA GPU and NVIDIA Container
Toolkit are required because GPU resource metrics are part of this benchmark.
EOF
}

log() {
  printf '\n==> %s\n' "$*"
}

info() {
  printf 'INFO: %s\n' "$*"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

compose_args() {
  local args=(-f "${COMPOSE_FILE}")
  if [[ -f "${ENV_FILE}" ]]; then
    args=(--env-file "${ENV_FILE}" "${args[@]}")
  fi
  printf '%s\n' "${args[@]}"
}

compose() {
  local args=()
  while IFS= read -r item; do
    args+=("${item}")
  done < <(compose_args)
  docker compose "${args[@]}" "$@"
}

check_docker() {
  command -v docker >/dev/null 2>&1 || die "Docker is not installed or not in PATH."
  docker version >/dev/null 2>&1 || die "The Docker daemon is not available."
  docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required."
  [[ -f "${COMPOSE_FILE}" ]] || die "Compose file not found: ${COMPOSE_FILE}"
}

check_required_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required."
  nvidia-smi -L >/dev/null 2>&1 || die "No usable NVIDIA GPU was detected."

  log "Testing NVIDIA GPU access from Docker."
  local output
  if ! output="$(
    docker run --rm --gpus all "${NVIDIA_TEST_IMAGE}" nvidia-smi 2>&1
  )"; then
    [[ -z "${output}" ]] || printf '%s\n' "${output}" | tail -n 8 >&2
    die "Docker cannot expose the NVIDIA GPU. Configure NVIDIA Container Toolkit."
  fi
  info "NVIDIA GPU access is available to Docker containers."
}

validate_compose() {
  compose config --quiet || die "The benchmark Compose configuration is invalid."
}

prepare_directories() {
  mkdir -p \
    "${PROJECT_ROOT}/outputs/ollama" \
    "${PROJECT_ROOT}/outputs/heracles_benchmark"
  [[ -w "${PROJECT_ROOT}/outputs/heracles_benchmark" ]] ||
    die "Benchmark output directory is not writable: ${PROJECT_ROOT}/outputs/heracles_benchmark"
}

check_container_name() {
  local owner_project
  if ! docker container inspect "${OLLAMA_CONTAINER_NAME}" >/dev/null 2>&1; then
    return
  fi
  owner_project="$(
    docker inspect \
      --format '{{ index .Config.Labels "com.docker.compose.project" }}' \
      "${OLLAMA_CONTAINER_NAME}" 2>/dev/null || true
  )"
  [[ "${owner_project}" == "${COMPOSE_PROJECT_NAME}" ]] ||
    die "Container ${OLLAMA_CONTAINER_NAME} already belongs to another Compose project."
}

ensure_benchmark_image() {
  if docker image inspect "${BENCHMARK_IMAGE}" >/dev/null 2>&1; then
    info "Reusing existing benchmark image: ${BENCHMARK_IMAGE}"
    return
  fi

  log "Benchmark image not found; building ${BENCHMARK_IMAGE}."
  compose build benchmark-ollama
}

force_build_benchmark_image() {
  log "Force-building the local Heracles benchmark image."
  compose build benchmark-ollama
  info "Benchmark image is ready: ${BENCHMARK_IMAGE}"
}

start_ollama() {
  check_container_name
  log "Starting GPU Ollama."
  compose up -d --wait ollama
  SERVICES_STARTED=1
}

start_services() {
  check_container_name
  ensure_benchmark_image
  log "Starting Neo4j, GPU Ollama, and the read-only Docker metrics proxy."
  compose up -d --wait neo4j ollama docker-socket-proxy
  SERVICES_STARTED=1
}

provision_models() {
  log "Validating and creating enabled local Ollama models."
  local records
  records="$(
    compose run \
      --rm \
      --no-deps \
      --entrypoint python \
      benchmark-ollama \
      /home/benchmark/workspace/model_manifest.py \
      --models-path "${MODEL_MANIFEST_IN_CONTAINER}" \
      --artifacts-root "${MODEL_ARTIFACTS_IN_CONTAINER}" \
      --format tsv
  )"

  local -a provisioning_records=()
  mapfile -t provisioning_records <<<"${records}"
  ((${#provisioning_records[@]} > 0)) ||
    die "The model manifest returned no enabled models."

  local model
  local modelfile
  local record
  local provisioned=0
  for record in "${provisioning_records[@]}"; do
    [[ "${record}" == *$'\t'* ]] ||
      die "Invalid model provisioning record: ${record}"
    model="${record%%$'\t'*}"
    modelfile="${record#*$'\t'}"
    [[ -n "${model}" && -n "${modelfile}" ]] ||
      die "Invalid model provisioning record: ${record}"
    info "Creating ${model} from ${modelfile}."
    compose exec -T ollama ollama create "${model}" -f "${modelfile}" </dev/null
    if ! compose exec -T ollama ollama show "${model}" </dev/null >/dev/null; then
      die "Ollama model verification failed after create: ${model}"
    fi
    ((provisioned += 1))
  done
  info "Provisioned and verified ${provisioned} enabled Ollama model(s)."
}

select_benchmarks() {
  cat <<'EOF'

Select benchmark questions:

  1) Cypher QA only
  2) Cypher-PDDL only
  3) Both
  q) Quit

EOF
  local selection
  while true; do
    read -r -p "Selection [1/2/3/q]: " selection
    case "${selection}" in
      1) set_benchmarks cypher; return ;;
      2) set_benchmarks pddl; return ;;
      3) set_benchmarks both; return ;;
      q | Q) exit 0 ;;
      *) printf 'Enter 1, 2, 3, or q.\n' >&2 ;;
    esac
  done
}

set_benchmarks() {
  case "$1" in
    cypher)
      RUN_CYPHER=1
      RUN_PDDL=0
      ;;
    pddl)
      RUN_CYPHER=0
      RUN_PDDL=1
      ;;
    both | cypher,pddl | pddl,cypher)
      RUN_CYPHER=1
      RUN_PDDL=1
      ;;
    *)
      die "Unknown benchmark selection '$1'; use cypher, pddl, or both."
      ;;
  esac
}

cleanup() {
  if ((CLEANED_UP == 0 && SERVICES_STARTED == 1 && KEEP_STACK == 0)); then
    CLEANED_UP=1
    log "Stopping benchmark services (the Ollama model cache is retained)."
    compose down --remove-orphans || true
  fi
}

on_interrupt() {
  cleanup
  exit 130
}

run_sweep() {
  trap cleanup EXIT
  trap on_interrupt INT TERM
  start_services
  provision_models
  log "Running the selected Ollama model sweep."
  compose run \
    --rm \
    --no-deps \
    benchmark-ollama \
    /home/benchmark/workspace/benchmark_runner.sh \
    "${RUN_CYPHER}" \
    "${RUN_PDDL}" \
    "${OUTPUT_DIR_IN_CONTAINER}"
  printf '\nBenchmark report:\n  %s/report.html\n' "${OUTPUT_DIR_ON_HOST}"
}

command="run"
benchmarks_explicit=0
if (($# > 0)); then
  case "$1" in
    run | build | models | ollama | status | down | config)
      command="$1"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
  esac
fi

if [[ "${command}" == "run" ]]; then
  while (($# > 0)); do
    case "$1" in
      --benchmarks)
        (($# >= 2)) || die "--benchmarks requires cypher, pddl, or both."
        set_benchmarks "$2"
        benchmarks_explicit=1
        shift 2
        ;;
      --keep-stack)
        KEEP_STACK=1
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        die "Unknown run option: $1"
        ;;
    esac
  done
  if ((benchmarks_explicit == 0)) && [[ -t 0 ]]; then
    select_benchmarks
  fi
fi

check_docker
case "${command}" in
  config)
    compose config
    ;;
  status)
    compose ps
    ;;
  down)
    compose down --remove-orphans
    ;;
  build)
    validate_compose
    force_build_benchmark_image
    ;;
  run | models | ollama)
    check_required_gpu
    validate_compose
    prepare_directories
    if [[ "${command}" == "run" ]]; then
      run_sweep
    elif [[ "${command}" == "models" ]]; then
      ensure_benchmark_image
      start_ollama
      provision_models
    else
      (($# > 0)) || die "Usage: run_benchmark.sh ollama COMMAND [ARGS...]"
      start_ollama
      compose exec -T ollama ollama "$@"
    fi
    ;;
esac
