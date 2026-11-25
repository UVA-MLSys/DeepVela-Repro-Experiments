#!/usr/bin/env bash
# Generic profiled run. The profile backend is supplied by profile_cmd.
set -u
CONF=${1:?usage: bash run_profile.sh workload.conf}
[ -f "$CONF" ] || { echo "missing config: $CONF" >&2; exit 2; }
source "$CONF"
run() { if [[ -n "${DRY_RUN:-}" ]]; then printf '[dry-run]'; printf ' %q' "$@"; echo; else "$@"; fi; }
for required in "${REQUIRED_FILES[@]}"; do
    [[ -e "$required" ]] || { echo "missing required file: $required" >&2; exit 2; }
done
run_id=${RUN_ID:-${JOB_ID:-manual}}
run_dir="$RESULTS_DIR/profile-$run_id"
[[ -n "${DRY_RUN:-}" ]] && run_dir="$(mktemp -d)/profile-$run_id"
mkdir -p "$run_dir"
cp "$CONF" "$run_dir/workload.conf"
cd "$run_dir"
setup_env
stage_inputs
app_env
app_cmd
profile_cmd
printf 'profile command:' | tee run_info.txt
printf ' %q' "${PROFILE_CMD[@]}" | tee -a run_info.txt
printf '\n' | tee -a run_info.txt
if [[ -n "${DRY_RUN:-}" ]]; then run bash -c '"$@"' -- "${PROFILE_CMD[@]}"; rc=0
else "${PROFILE_CMD[@]}" >profile.out 2>&1; rc=$?; fi
echo "exit_status=$rc" | tee -a run_info.txt
exit "$rc"
