#!/usr/bin/env bash
# Run all validation cases against a sam3-server.
#
# Usage:
#   bash scripts/validate_cases.sh <server_url> [cases_dir] [--threshold 0.9]
#
# cases_dir defaults to tests/cases. Every subdirectory containing case.json
# is validated via scripts/validate_case.py. Exit code 0 iff all cases PASS.

set -u

SERVER_URL="${1:?usage: validate_cases.sh <server_url> [cases_dir] [--threshold 0.9]}"
CASES_DIR="${2:-tests/cases}"
THRESHOLD="${3:-}"

if [ -n "$THRESHOLD" ] && [ "$THRESHOLD" != "--threshold" ]; then
    echo "error: third argument must be --threshold" >&2
    exit 2
fi

if [ ! -d "$CASES_DIR" ]; then
    echo "error: cases dir not found: $CASES_DIR" >&2
    exit 2
fi

n_total=0
n_pass=0
failed=()

PY_RUNNER=""
if command -v uv >/dev/null 2>&1; then
    PY_RUNNER="uv run python"
elif command -v python3 >/dev/null 2>&1 && python3 -c "import PIL, numpy" >/dev/null 2>&1; then
    PY_RUNNER="python3"
else
    echo "ERROR: neither 'uv' nor python3 with PIL/numpy is available" >&2
    exit 2
fi

for case_dir in "$CASES_DIR"/*/; do
    [ -f "$case_dir/case.json" ] || continue
    n_total=$((n_total + 1))
    echo "=== case: $case_dir ==="
    if $PY_RUNNER scripts/validate_case.py "$SERVER_URL" "$case_dir"; then
        n_pass=$((n_pass + 1))
    else
        failed+=("$case_dir")
    fi
done

echo
echo "TOTAL: $n_pass/$n_total PASS"
if [ "$n_pass" -ne "$n_total" ]; then
    echo "FAILED CASES:"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
