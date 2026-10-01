#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/docker-compose.yaml}"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
NVIDIA_TEST_IMAGE="${NVIDIA_TEST_IMAGE:-ubuntu:24.04}"
DEMO_DATA_CONFIG="${DEMO_DATA_CONFIG:-configs/data.yaml}"
DEMO_TEACHER_CONFIG="${DEMO_TEACHER_CONFIG:-configs/teacher_synthetic.yaml}"
DEMO_STUDENT_CONFIG="${DEMO_STUDENT_CONFIG:-configs/student_qwen3.5_4b.yaml}"

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"

usage() {
  cat <<'EOF'
Usage: docker/stack.sh COMMAND [ARGS...]

Recommended first run:
  1. cp docker/.env.example docker/.env and set required credentials.
  2. docker/stack.sh setup
  3. docker/stack.sh demo-dataset
  4. docker/stack.sh demo-train
  5. docker/stack.sh demo-validate
  6. docker/stack.sh demo-deploy

Setup and lifecycle:
  setup                    Validate, build, and start the complete stack.
  check                    Validate Docker, Compose, and NVIDIA GPU support.
  build                    Rebuild the main project image.
  up                       Start the main project container and Neo4j.
  ollama-up                Start the optional Ollama GGUF inference container.
  down                     Stop the stack; persistent volumes are retained.
  restart [SERVICE...]     Restart the complete stack or selected services.

Main project container:
  main COMMAND [ARGS...]   Run a command in the running main container.
  shell                    Open an interactive shell in the main container.
  exec COMMAND [ARGS...]   Alias for main (kept for compatibility).
  ollama COMMAND [ARGS...] Run the Ollama CLI in the optional container.

Automated workflow:
  demo-dataset             Create a dataset, teacher traces, and grouped splits.
  demo-train               Train a new student model using the current dataset.
  demo-validate            Evaluate the student and baseline and generate HTML reports.
  demo-deploy              Export the trained student to GGUF, register the
                           configured variants in Ollama, validate them on
                           the challenge split, and regenerate the HTML report.

Neo4j:
  load-neo4j [SCENE]       Load a scene into the running Neo4j service.

Inspection:
  status                   Show service status.
  logs [SERVICE...]        Follow logs for all services or selected services.
  config                   Validate and print the merged Compose config.
  ps                       Alias for status (kept for compatibility).

Examples:
  docker/stack.sh main python scripts/training/train_agent_lora.py \
    --data-config configs/data.yaml \
    --student-config configs/student_qwen3.5_4b.yaml
  docker/stack.sh logs neo4j

Environment:
  ENV_FILE=path            Optional Compose env file. Defaults to docker/.env.
  DEMO_DATA_CONFIG=path    Demo data config. Defaults to configs/data.yaml.
  DEMO_TEACHER_CONFIG=path Demo teacher config. Defaults to the synthetic teacher.
  DEMO_STUDENT_CONFIG=path Demo student config. Defaults to Qwen 3.5 4B.
EOF
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

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

die() {
  printf '\n[ERROR] %s\n' "$*" >&2
  exit 1
}

heading() {
  local title="$1"
  printf '\n============================================================\n'
  printf '  %s\n' "${title}"
  printf '============================================================\n'
}

demo_step() {
  local current="$1"
  local total="$2"
  local title="$3"
  shift 3

  heading "Step ${current}/${total}  ${title}"
  if ! compose exec main "$@"; then
    die "Step ${current}/${total} failed: ${title}"
  fi
}

workflow_step() {
  local current="$1"
  local total="$2"
  local title="$3"
  shift 3

  heading "Step ${current}/${total}  ${title}"
  if ! "$@"; then
    die "Step ${current}/${total} failed: ${title}"
  fi
}

render_group_progress() {
  local title="$1"
  local current="$2"
  local total="$3"
  local elapsed="$4"
  local width=32
  local filled=$((current * width / total))
  local empty=$((width - filled))
  local bar=""
  local index
  local padding

  for ((index = 0; index < filled; index++)); do
    bar+="━"
  done
  printf -v padding '%*s' "${empty}" ''
  bar+="${padding}"
  if [[ -t 1 ]]; then
    printf '\r\033[2K%-38s %s %d/%d %d:%02d' \
      "${title}" "${bar}" "${current}" "${total}" \
      "$((elapsed / 60))" "$((elapsed % 60))"
    if [[ "${current}" -eq "${total}" ]]; then
      printf '\n'
    fi
    return
  fi

  printf '%-38s %s %d/%d %d:%02d\n' \
    "${title}" "${bar}" "${current}" "${total}" \
    "$((elapsed / 60))" "$((elapsed % 60))"
}

quiet_phase() {
  local current="$1"
  local total="$2"
  local started_at="$3"
  local title="$4"
  local output_file
  shift 4

  output_file="$(mktemp "${TMPDIR:-/tmp}/3dsg-deploy.XXXXXX")"
  if [[ -t 1 ]]; then
    render_group_progress "${title}" "$((current - 1))" "${total}" \
      "$((SECONDS - started_at))"
  fi
  if "$@" >"${output_file}" 2>&1; then
    rm -f "${output_file}"
    render_group_progress "${title}" "${current}" "${total}" \
      "$((SECONDS - started_at))"
    return 0
  fi

  if [[ -t 1 ]]; then
    printf '\n'
  fi
  warn "Captured command output (last 80 lines):"
  tr '\r' '\n' <"${output_file}" | tail -n 80 >&2
  rm -f "${output_file}"
  return 1
}

register_ollama_models() {
  local model_name="$1"
  local model_dir="$2"
  shift 2
  local variants=("$@")
  local total=$((1 + ${#variants[@]}))
  local current=1
  local variant
  local started_at="${SECONDS}"

  quiet_phase "${current}" "${total}" "${started_at}" "Starting Ollama" \
    compose --profile ollama up -d --wait ollama || return 1
  for variant in "${variants[@]}"; do
    current=$((current + 1))
    quiet_phase "${current}" "${total}" "${started_at}" "Registering ${variant} model" \
      compose exec -T ollama ollama create "${model_name}-${variant,,}" \
        -f "${model_dir}/Modelfile.${variant,,}" || return 1
  done

  compose exec -T main python scripts/training/deploy_helpers.py registration-summary \
    --model-name "${model_name}" --variants "${variants[@]}"
}

check_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'EOF'

ERROR: Docker is not installed or the docker command is not in PATH.

Install Docker Engine or Docker Desktop before using this stack.

EOF
    exit 1
  fi

  if ! docker version >/dev/null 2>&1; then
    cat >&2 <<'EOF'

ERROR: The Docker client is installed, but it cannot communicate with the
Docker daemon.

Possible causes include:

  - Docker is not running.
  - The current user does not have permission to access Docker.
  - DOCKER_HOST points to an unavailable daemon.

Verify the installation with:

  docker info

EOF
    exit 1
  fi

  if ! docker compose version >/dev/null 2>&1; then
    cat >&2 <<'EOF'

ERROR: Docker Compose v2 is not available.

This script requires the `docker compose` plugin rather than the legacy
`docker-compose` command.

EOF
    exit 1
  fi

  info "Docker and Docker Compose are available."
}

host_has_nvidia_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 &&
    nvidia-smi -L >/dev/null 2>&1
}

nvidia_toolkit_command_exists() {
  command -v nvidia-ctk >/dev/null 2>&1 ||
    command -v nvidia-container-cli >/dev/null 2>&1
}

docker_reports_nvidia_runtime() {
  docker info --format '{{json .Runtimes}}' 2>/dev/null |
    grep -qi 'nvidia'
}

test_docker_gpu_access() {
  local output

  info "Testing NVIDIA GPU access from Docker."

  if output="$(
    docker run \
      --rm \
      --gpus all \
      "${NVIDIA_TEST_IMAGE}" \
      nvidia-smi 2>&1
  )"; then
    return 0
  fi

  warn "Docker could not create an NVIDIA GPU-enabled test container."

  if [[ -n "${output}" ]]; then
    printf '%s\n' "${output}" |
      tail -n 8 |
      sed 's/^/  /' >&2
  fi

  return 1
}

report_gpu_memory() {
  local max_vram_mib

  max_vram_mib="$(
    nvidia-smi \
      --query-gpu=memory.total \
      --format=csv,noheader,nounits 2>/dev/null |
      tr -d ' ' |
      sort -nr |
      head -n 1
  )"

  if [[ "${max_vram_mib}" =~ ^[0-9]+$ ]]; then
    info "Largest detected NVIDIA GPU: $((max_vram_mib / 1024)) GiB VRAM."
  else
    warn "Could not determine NVIDIA GPU memory."
  fi
}

check_required_gpu() {
  if ! host_has_nvidia_gpu; then
    cat >&2 <<'EOF'

ERROR: No usable NVIDIA GPU was detected.

This Docker setup requires GPU access because the configured student training
uses QLoRA (`load_in_4bit: true`), bitsandbytes, and bf16 CUDA execution.

Verify the host GPU with:

  nvidia-smi

EOF
    exit 1
  fi

  info "An NVIDIA GPU was detected."
  report_gpu_memory

  if nvidia_toolkit_command_exists; then
    info "NVIDIA Container Toolkit commands were found."
  elif docker_reports_nvidia_runtime; then
    info "Docker reports an NVIDIA container runtime."
  else
    cat >&2 <<'EOF'

ERROR: The NVIDIA Container Toolkit could not be identified.

Install and configure NVIDIA Container Toolkit for Docker. A typical
configuration is:

  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker

EOF
    exit 1
  fi

  if ! test_docker_gpu_access; then
    cat >&2 <<'EOF'

ERROR: Docker cannot expose the NVIDIA GPU to containers.

Configure NVIDIA Container Toolkit for Docker, restart Docker, and retry:

  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker

EOF
    exit 1
  fi

  info "Docker can expose NVIDIA GPUs to containers."
}

set_host_user_ids() {
  if ! command -v id >/dev/null 2>&1; then
    die "The id command is required to determine the host UID and GID."
  fi

  export HOST_UID
  export HOST_GID

  HOST_UID="$(id -u)"
  HOST_GID="$(id -g)"

  if [[ "${HOST_UID}" == "0" || "${HOST_GID}" == "0" ]]; then
    warn "The main image is being built using root UID or GID."
  fi

  info "Main container UID:GID will be ${HOST_UID}:${HOST_GID}."
}

validate_compose_file() {
  [[ -f "${COMPOSE_FILE}" ]] ||
    die "Compose file not found: ${COMPOSE_FILE}"

  if ! compose config --quiet; then
    die "The Docker Compose configuration is invalid."
  fi
}

preflight() {
  check_docker
  set_host_user_ids
  check_required_gpu
  validate_compose_file
}

prepare_runtime() {
  check_docker
  set_host_user_ids
  validate_compose_file
}

require_service_running() {
  local service="$1"

  if compose ps --services --status running | grep -Fxq "${service}"; then
    return
  fi

  die "Service '${service}' is not running. Run 'docker/stack.sh setup' on the first run or 'docker/stack.sh up' to start the stack."
}

run_main() {
  if [[ $# -lt 1 ]]; then
    die "Missing main-container command. Usage: docker/stack.sh main COMMAND [ARGS...]"
  fi

  prepare_runtime
  require_service_running main
  compose exec main "$@"
}

run_ollama() {
  if [[ $# -lt 1 ]]; then
    die "Missing Ollama command. Usage: docker/stack.sh ollama COMMAND [ARGS...]"
  fi

  prepare_runtime
  require_service_running ollama
  compose exec -T ollama ollama "$@"
}

run_demo_dataset() {
  local total_steps=5

  if [[ $# -ne 0 ]]; then
    die "The demo-dataset command takes no arguments. Configure it with the DEMO_* environment variables."
  fi

  prepare_runtime
  require_service_running main
  require_service_running neo4j

  heading "3DSG Cypher agent demo: dataset"
  info "Teacher config: ${DEMO_TEACHER_CONFIG}"
  info "Data config: ${DEMO_DATA_CONFIG}"
  info "Student config: ${DEMO_STUDENT_CONFIG}"

  demo_step 1 "${total_steps}" "Extract scene template values" \
    python scripts/graph/extract_3dsg_values.py \
      --data-config "${DEMO_DATA_CONFIG}"
  demo_step 2 "${total_steps}" "Generate question templates" \
    python scripts/cypher/generate_3dsg_seed_templates.py \
      --data-config "${DEMO_DATA_CONFIG}"
  demo_step 3 "${total_steps}" "Prepare seed records" \
    python scripts/agents/prepare_dataset.py prepare-seeds \
      --data-config "${DEMO_DATA_CONFIG}"
  demo_step 4 "${total_steps}" "Generate and validate teacher traces" \
    python scripts/agents/generate_traces.py generate \
      --data-config "${DEMO_DATA_CONFIG}" \
      --teacher-config "${DEMO_TEACHER_CONFIG}" \
      --student-config "${DEMO_STUDENT_CONFIG}"
  demo_step 5 "${total_steps}" "Create grouped dataset splits" \
    python scripts/agents/prepare_dataset.py split \
      --data-config "${DEMO_DATA_CONFIG}" \
      --render-view ollama_native

  heading "Demo dataset complete"
}

run_demo_train() {
  local total_steps=1

  if [[ $# -ne 0 ]]; then
    die "The demo-train command takes no arguments. Configure it with the DEMO_* environment variables."
  fi

  prepare_runtime
  require_service_running main

  heading "3DSG Cypher agent demo: train"
  info "Data config: ${DEMO_DATA_CONFIG}"
  info "Student config: ${DEMO_STUDENT_CONFIG}"

  demo_step 1 "${total_steps}" "Train student model with LoRA" \
    python scripts/training/train_agent_lora.py \
      --data-config "${DEMO_DATA_CONFIG}" \
      --student-config "${DEMO_STUDENT_CONFIG}"

  heading "Demo train complete"
}

run_demo_validate() {
  local total_steps=6

  if [[ $# -ne 0 ]]; then
    die "The demo-validate command takes no arguments. Configure it with the DEMO_* environment variables."
  fi

  prepare_runtime
  require_service_running main
  require_service_running neo4j

  heading "3DSG Cypher agent demo: validate"
  info "Data config: ${DEMO_DATA_CONFIG}"
  info "Student config: ${DEMO_STUDENT_CONFIG}"

  demo_step 1 "${total_steps}" "Evaluate loss on all splits" \
    python scripts/agents/evaluate_student.py \
      --data-config "${DEMO_DATA_CONFIG}" \
      --student-config "${DEMO_STUDENT_CONFIG}" \
      --split all \
      --mode eval-loss
  demo_step 2 "${total_steps}" "Validate generation on the test split" \
    python scripts/agents/evaluate_student.py \
      --data-config "${DEMO_DATA_CONFIG}" \
      --student-config "${DEMO_STUDENT_CONFIG}" \
      --split test \
      --mode raw-generation \
      --max-new-tokens 1024
  demo_step 3 "${total_steps}" "Validate generation on the challenge split" \
    python scripts/agents/evaluate_student.py \
      --data-config "${DEMO_DATA_CONFIG}" \
      --student-config "${DEMO_STUDENT_CONFIG}" \
      --split challenge \
      --mode raw-generation \
      --max-new-tokens 1024
  demo_step 4 "${total_steps}" "Establish baseline on the test split" \
    python scripts/agents/evaluate_student.py \
      --data-config "${DEMO_DATA_CONFIG}" \
      --student-config "${DEMO_STUDENT_CONFIG}" \
      --split test \
      --mode raw-generation \
      --baseline \
      --max-new-tokens 1024
  demo_step 5 "${total_steps}" "Establish baseline on the challenge split" \
    python scripts/agents/evaluate_student.py \
      --data-config "${DEMO_DATA_CONFIG}" \
      --student-config "${DEMO_STUDENT_CONFIG}" \
      --split challenge \
      --mode raw-generation \
      --baseline \
      --max-new-tokens 1024
  demo_step 6 "${total_steps}" "Generate HTML workflow report" \
    python scripts/reporting/generate_workflow_report.py \
      --data-config "${DEMO_DATA_CONFIG}" \
      --student-config "${DEMO_STUDENT_CONFIG}"

  heading "Demo validate complete"
}

require_trained_student() {
  if compose exec -T main python scripts/training/deploy_helpers.py check \
    --student-config "${DEMO_STUDENT_CONFIG}"; then
    return
  fi

  die "The trained student configured by '${DEMO_STUDENT_CONFIG}' is unavailable. Run 'docker/stack.sh demo-train' or train the configured student before running demo-deploy."
}

run_demo_deploy() {
  local total_steps
  local model_name
  local ollama_model_dir
  local deploy_config
  local config_lines=()
  local variants=()
  local variant
  local current_step

  if [[ $# -ne 0 ]]; then
    die "The demo-deploy command takes no arguments. Configure it with the DEMO_* environment variables."
  fi

  prepare_runtime
  require_service_running main
  require_service_running neo4j
  require_trained_student

  if ! deploy_config="$(compose exec -T main python scripts/training/deploy_helpers.py config \
    --student-config "${DEMO_STUDENT_CONFIG}")"; then
    die "Could not read deployment settings from '${DEMO_STUDENT_CONFIG}'."
  fi
  mapfile -t config_lines <<< "${deploy_config}"
  model_name="${config_lines[0]}"
  variants=("${config_lines[@]:1}")
  [[ -n "${model_name}" && ${#variants[@]} -gt 0 ]] || die "Deployment settings contain no model or variants."
  ollama_model_dir="/models/${model_name}"
  total_steps=$((4 + ${#variants[@]}))

  heading "3DSG Cypher agent deploy / quantization pipeline"
  info "Data config: ${DEMO_DATA_CONFIG}"
  info "Student config: ${DEMO_STUDENT_CONFIG}"
  info "Exported model name: ${model_name}"

  workflow_step 1 "${total_steps}" "Export merged model and GGUF quantizations" \
    compose exec main python scripts/training/export_agent_ollama.py \
      --student-config "${DEMO_STUDENT_CONFIG}" \
      --model-name "${model_name}" \
      --quiet \
      --stages merge convert quantize
  workflow_step 2 "${total_steps}" "Generate Ollama Modelfiles" \
    compose exec main python scripts/training/export_agent_ollama.py \
      --student-config "${DEMO_STUDENT_CONFIG}" \
      --model-name "${model_name}" \
      --quiet \
      --stages modelfile \
      --modelfile-variants "${variants[@]}"
  workflow_step 3 "${total_steps}" "Start Ollama and register GGUF variants" \
    register_ollama_models "${model_name}" "${ollama_model_dir}" "${variants[@]}"
  current_step=4
  for variant in "${variants[@]}"; do
    workflow_step "${current_step}" "${total_steps}" "Validate ${variant} generation on the challenge split" \
      compose exec main python scripts/agents/evaluate_student.py \
        --data-config "${DEMO_DATA_CONFIG}" \
        --student-config "${DEMO_STUDENT_CONFIG}" \
        --split challenge \
        --mode raw-generation \
        --runtime ollama \
        --ollama-model "${model_name}-${variant,,}" \
        --max-new-tokens 1024
    current_step=$((current_step + 1))
  done
  workflow_step "${current_step}" "${total_steps}" "Generate HTML workflow report" \
    compose exec main python scripts/reporting/generate_workflow_report.py \
      --data-config "${DEMO_DATA_CONFIG}" \
      --student-config "${DEMO_STUDENT_CONFIG}"

  heading "Deploy / quantization pipeline complete"
  info "The HTML report path is shown in the final summary above."
}

if [[ $# -lt 1 ]]; then
  usage
  exit 0
fi

command="$1"
shift

if [[ "${command}" == "help" || "${command}" == "-h" || "${command}" == "--help" ]]; then
  usage
  exit 0
fi

case "${command}" in
  setup)
    preflight
    compose build main
    compose up -d neo4j main
    info "Stack setup is complete. Next: docker/stack.sh demo-dataset"
    ;;
  check)
    preflight
    ;;
  build)
    preflight
    compose build main
    ;;
  up)
    preflight
    compose up -d neo4j main
    ;;
  ollama-up)
    preflight
    compose --profile ollama up -d --wait ollama
    ;;
  down)
    check_docker
    compose --profile ollama down --remove-orphans
    ;;
  restart)
    prepare_runtime
    compose restart "$@"
    ;;
  logs)
    check_docker
    compose logs -f "$@"
    ;;
  status | ps)
    check_docker
    compose ps
    ;;
  shell)
    prepare_runtime
    require_service_running main
    compose exec main bash
    ;;
  main | exec)
    run_main "$@"
    ;;
  ollama)
    run_ollama "$@"
    ;;
  demo-dataset)
    run_demo_dataset "$@"
    ;;
  demo-validate)
    run_demo_validate "$@"
    ;;
  demo-train)
    run_demo_train "$@"
    ;;
  demo-deploy)
    run_demo_deploy "$@"
    ;;
  load-neo4j)
    scene="${1:-data/scenes/uhumans2_office.json}"
    prepare_runtime
    require_service_running main
    require_service_running neo4j
    compose exec main python scripts/graph/load_3dsg_neo4j.py --scene "${scene}"
    ;;
  config)
    prepare_runtime
    compose config
    ;;
  *)
    usage
    exit 2
    ;;
esac
