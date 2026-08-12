#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster/euler_account.sh
source "$ROOT/euler_account.sh"

unset TERRA_EULER_USER TERRA_EULER_HOME_ROOT
unset TERRA_EULER_SCRATCH_ROOT TERRA_EULER_PROJECT_ROOT
terra_euler_configure alesweber
test "$TERRA_EULER_USER" = alesweber
test "$TERRA_EULER_HOME_ROOT" = /cluster/home/alesweber
test "$TERRA_EULER_SCRATCH_ROOT" = /cluster/scratch/alesweber
test "$TERRA_EULER_PROJECT_ROOT" = /cluster/project/rsl/alesweber

unset TERRA_EULER_USER TERRA_EULER_HOME_ROOT
unset TERRA_EULER_SCRATCH_ROOT TERRA_EULER_PROJECT_ROOT
TERRA_EULER_SCRATCH_ROOT=/cluster/scratch/shared-terra
terra_euler_configure lterenzi
test "$TERRA_EULER_USER" = lterenzi
test "$TERRA_EULER_HOME_ROOT" = /cluster/home/lterenzi
test "$TERRA_EULER_SCRATCH_ROOT" = /cluster/scratch/shared-terra
test "$TERRA_EULER_PROJECT_ROOT" = /cluster/project/rsl/lterenzi

unset TERRA_EULER_USER TERRA_EULER_HOME_ROOT
unset TERRA_EULER_SCRATCH_ROOT TERRA_EULER_PROJECT_ROOT
if terra_euler_configure 'bad/account'; then
    echo "invalid account unexpectedly accepted" >&2
    exit 1
fi

unset TERRA_EULER_USER TERRA_EULER_HOME_ROOT
unset TERRA_EULER_SCRATCH_ROOT TERRA_EULER_PROJECT_ROOT
if terra_euler_configure ''; then
    echo "empty account unexpectedly accepted" >&2
    exit 1
fi

unset TERRA_EULER_USER TERRA_EULER_HOME_ROOT
unset TERRA_EULER_SCRATCH_ROOT TERRA_EULER_PROJECT_ROOT
TERRA_EULER_HOME_ROOT=/tmp/not-euler
if terra_euler_configure alesweber; then
    echo "non-cluster storage root unexpectedly accepted" >&2
    exit 1
fi
test -z "${TERRA_EULER_USER:-}"

unset TERRA_EULER_USER TERRA_EULER_HOME_ROOT
unset TERRA_EULER_SCRATCH_ROOT TERRA_EULER_PROJECT_ROOT
TERRA_EULER_PROJECT_ROOT="/cluster/project/rsl/bad'path"
if terra_euler_configure alesweber; then
    echo "unsafe storage root unexpectedly accepted" >&2
    exit 1
fi
test -z "${TERRA_EULER_USER:-}"

echo EULER_ACCOUNT_TESTS_PASSED
