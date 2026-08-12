#!/usr/bin/env bash

# Resolve account-owned Euler storage without coupling a launcher to one user.
# Source this file and call terra_euler_configure [account]. Callers may
# override any TERRA_EULER_*_ROOT before invoking the function.
terra_euler_configure() {
    local account="${1-${TERRA_EULER_USER:-}}"

    case "$account" in
        ''|*[!a-zA-Z0-9_-]*)
            echo "invalid or missing Euler account '$account'" >&2
            return 2
            ;;
        *) ;;
    esac

    local home_root="${TERRA_EULER_HOME_ROOT:-/cluster/home/$account}"
    local scratch_root="${TERRA_EULER_SCRATCH_ROOT:-/cluster/scratch/$account}"
    local project_root="${TERRA_EULER_PROJECT_ROOT:-/cluster/project/rsl/$account}"

    local path
    for path in \
        "$home_root" \
        "$scratch_root" \
        "$project_root"; do
        case "$path" in
            /cluster/*) ;;
            *)
                echo "Euler storage root must be absolute under /cluster: $path" >&2
                return 2
                ;;
        esac
        case "$path" in
            *[!a-zA-Z0-9_./-]*)
                echo "Euler storage root contains unsupported characters: $path" >&2
                return 2
                ;;
        esac
    done

    TERRA_EULER_USER="$account"
    TERRA_EULER_HOME_ROOT="$home_root"
    TERRA_EULER_SCRATCH_ROOT="$scratch_root"
    TERRA_EULER_PROJECT_ROOT="$project_root"
    export TERRA_EULER_USER TERRA_EULER_HOME_ROOT
    export TERRA_EULER_SCRATCH_ROOT TERRA_EULER_PROJECT_ROOT
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    terra_euler_configure "${1-}"
    printf '%s\n' \
        "TERRA_EULER_USER=$TERRA_EULER_USER" \
        "TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT" \
        "TERRA_EULER_SCRATCH_ROOT=$TERRA_EULER_SCRATCH_ROOT" \
        "TERRA_EULER_PROJECT_ROOT=$TERRA_EULER_PROJECT_ROOT"
fi
