#!/usr/bin/env bash
set -Eeuo pipefail

RUN_CYPHER_QA="${1:-1}"
RUN_CYPHER_PDDL="${2:-1}"
OUTPUT_DIR="${3:-output/model_sweep}"
WORKSPACE_DIR="/home/benchmark/workspace"
OVERALL_STATUS=0
RUN_MARKER="$(mktemp)"

cleanup() {
  rm -f "${RUN_MARKER}"
}

trap cleanup EXIT

cd "${WORKSPACE_DIR}/heracles_agents"

log() {
  printf '\n==> %s\n' "$*"
}

record_failure() {
  printf '\nERROR: %s\n' "$1" >&2
  OVERALL_STATUS=1
}

validate_flag() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    printf 'ERROR: %s must be 0 or 1; got %q.\n' "${name}" "${value}" >&2
    exit 2
  fi
}

validate_flag RUN_CYPHER_QA "${RUN_CYPHER_QA}"
validate_flag RUN_CYPHER_PDDL "${RUN_CYPHER_PDDL}"
if [[ "${RUN_CYPHER_QA}" == "0" && "${RUN_CYPHER_PDDL}" == "0" ]]; then
  printf 'ERROR: At least one benchmark must be selected.\n' >&2
  exit 2
fi

log "Loading the canonical Heracles scene graph into Neo4j."
if ! python "${WORKSPACE_DIR}/heracles/examples/load_scene_graph.py" \
  --scene_graph "${WORKSPACE_DIR}/heracles/examples/scene_graphs/example_dsg.json"; then
  record_failure "Failed to load the Neo4j database."
else
  experiments=()
  if [[ "${RUN_CYPHER_QA}" == "1" ]]; then
    experiments+=(examples/experiments/ollama/cypher_qa_student_model_sweep.yaml)
  fi
  if [[ "${RUN_CYPHER_PDDL}" == "1" ]]; then
    experiments+=(examples/experiments/ollama/cypher_pddl_student_model_sweep.yaml)
  fi

  log "Running the selected Ollama student model sweep."
  if ! python examples/experiment_runner.py \
    "${experiments[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --continue-on-error \
    --no-display \
    --debug; then
    record_failure "The selected Ollama experiment failed."
  fi
fi

log "Generating the static HTML report."
result_directories=()
if [[ "${RUN_CYPHER_QA}" == "1" ]]; then
  result_directories+=("${OUTPUT_DIR}/ollama/cypher_qa_student_model_sweep")
fi
if [[ "${RUN_CYPHER_PDDL}" == "1" ]]; then
  result_directories+=("${OUTPUT_DIR}/ollama/cypher_pddl_student_model_sweep")
fi

result_files=()
for directory in "${result_directories[@]}"; do
  [[ -d "${directory}" ]] || continue
  while IFS= read -r -d '' result_file; do
    result_files+=("${result_file}")
  done < <(
    find "${directory}" \
      -maxdepth 1 \
      -type f \
      -name '*_results.yaml' \
      -newer "${RUN_MARKER}" \
      -print0 | sort -z
  )
done

if ((${#result_files[@]} == 0)); then
  record_failure "No result YAML files were found; report.html was not generated."
elif ! python examples/display_yaml_results.py \
  "${result_files[@]}" \
  --mode html \
  --output "${OUTPUT_DIR}/report.html"; then
  record_failure "Failed to generate report.html."
else
  printf '\nReport generated at:\n  %s/heracles_agents/%s/report.html\n' \
    "${WORKSPACE_DIR}" "${OUTPUT_DIR}"
fi

exit "${OVERALL_STATUS}"
