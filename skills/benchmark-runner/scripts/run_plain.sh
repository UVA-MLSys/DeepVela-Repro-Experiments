#!/usr/bin/env bash
# Generic plain workload run. A scheduler, if used, launches this script.
set -u
CONF=${1:?usage: bash run_plain.sh workload.conf}
[ -f "$CONF" ] || { echo "missing config: $CONF" >&2; exit 2; }
source "$CONF"
run() { if [[ -n "${DRY_RUN:-}" ]]; then printf '[dry-run]'; printf ' %q' "$@"; echo; else "$@"; fi; }
for required in "${REQUIRED_FILES[@]}"; do
    [[ -e "$required" ]] || { echo "missing required file: $required" >&2; exit 2; }
done
run_id=${RUN_ID:-${JOB_ID:-manual}}
run_dir="$RESULTS_DIR/plain-$run_id"
[[ -n "${DRY_RUN:-}" ]] && run_dir="$(mktemp -d)/plain-$run_id"
mkdir -p "$run_dir"
cp "$CONF" "$run_dir/workload.conf"
cd "$run_dir"
setup_env
stage_inputs
app_env
app_cmd
printf 'command:' | tee run_info.txt
printf ' %q' "${APP_CMD[@]}" | tee -a run_info.txt
printf '\n' | tee -a run_info.txt
if [[ -n "${DRY_RUN:-}" ]]; then run bash -c '"$@"' -- "${APP_CMD[@]}"; rc=0
else "${APP_CMD[@]}" >app.out 2>&1; rc=$?; fi
echo "exit_status=$rc" | tee -a run_info.txt
if [[ -n "${PERF_GREP:-}" && -f app.out ]]; then grep -E "$PERF_GREP" app.out | tail -n 3 || true; fi
exit "$rc"
