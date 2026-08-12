#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: lquota_home_used_gb.sh /cluster/home/<account>" >&2
    exit 2
fi
HOME_ROOT="$1"
case "$HOME_ROOT" in
    /cluster/home/*) ;;
    *) echo "invalid Euler home root: $HOME_ROOT" >&2; exit 2 ;;
esac

awk -F'|' -v expected_home="$HOME_ROOT" '
function trim(value) {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
    return value
}
$2 ~ /\/cluster\/home\// {
    home = trim($2)
    quota_type = trim($3)
    if (home == expected_home && quota_type == "space") {
        matches += 1
        used = trim($4)
        fields = split(used, parts, /[[:space:]]+/)
        if (fields != 2 || parts[1] !~ /^[0-9]+([.][0-9]+)?$/) {
            invalid = 1
            next
        }
        value = parts[1] + 0
        unit = parts[2]
        if (unit == "kB") factor = 0.000001
        else if (unit == "MB") factor = 0.001
        else if (unit == "GB") factor = 1
        else if (unit == "TB") factor = 1000
        else {
            invalid = 1
            next
        }
        used_gb = value * factor
    }
}
END {
    if (matches != 1 || invalid) exit 2
    printf "%.6f\n", used_gb
}
'
