#!/usr/bin/env bash
set -euo pipefail
CONF=${1:?usage: make_runbook.sh workload.conf}
source "$CONF"
out="${RUNBOOK_OUT:-RUNBOOK_${WORKLOAD}.md}"
cat >"$out" <<EOF
# Runbook: ${WORKLOAD}

${WORKLOAD_DESC}

The plain run is the canonical timing/correctness run. The profile run is optional and must not
replace it. Launch either script with the scheduler or process manager appropriate to the environment.

## 1. Validate without launching

\`\`\`bash
DRY_RUN=1 bash run_plain.sh ${CONF}
DRY_RUN=1 bash run_profile.sh ${CONF}
\`\`\`

## 2. Canonical run

\`\`\`bash
bash run_plain.sh ${CONF}
\`\`\`

Check exit status, application correctness, sample count, and the application-owned timing output.

## 3. Optional profile run

\`\`\`bash
bash run_profile.sh ${CONF}
\`\`\`

Check the profiler's own output and preserve its version, configuration, and overhead separately.
EOF
echo "$out"
