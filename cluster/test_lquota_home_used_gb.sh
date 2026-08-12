#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER="$ROOT/lquota_home_used_gb.sh"
HOME_ROOT=/cluster/home/alesweber

row() {
    printf '| %s | space | %s | 45.00 GB | 50.00 GB |\n' "$HOME_ROOT" "$1"
}

test "$(row '37.36 GB' | "$PARSER" "$HOME_ROOT")" = 37.360000
test "$(row '37360 MB' | "$PARSER" "$HOME_ROOT")" = 37.360000
test "$(row '0.03736 TB' | "$PARSER" "$HOME_ROOT")" = 37.360000
test "$(row '37360000 kB' | "$PARSER" "$HOME_ROOT")" = 37.360000

if row '37.36 GiB' | "$PARSER" "$HOME_ROOT"; then
    echo "unsupported quota unit unexpectedly accepted" >&2
    exit 1
fi
if { row '37.36 GB'; row '37.36 GB'; } | "$PARSER" "$HOME_ROOT"; then
    echo "duplicate quota rows unexpectedly accepted" >&2
    exit 1
fi
if row '37.36 GB' | "$PARSER" /cluster/home/lterenzi; then
    echo "wrong home row unexpectedly accepted" >&2
    exit 1
fi

echo LQUOTA_HOME_USED_GB_TESTS_PASSED
