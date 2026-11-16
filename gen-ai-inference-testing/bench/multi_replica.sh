#!/usr/bin/env bash
# Load N replicas at the same time, each with its own client and results dir,
# to check that the per-replica SLO survives instance-level contention.
#   N=4 SHAPES="1024:128 4096:256" CONCURRENCIES="1 8 16" bench/multi_replica.sh <run-prefix>
# Then: python3 bench/compare_replicas.py --baseline <single-replica run> \
#         --replica-dirs <prefix>-r$N-rep0 ... <prefix>-r$N-rep$((N-1))
set -uo pipefail
PREFIX=$1; N="${N:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for k in $(seq 0 $((N - 1))); do
  TARGET=$k bash "$HERE/client.sh" vllm "${PREFIX}-r${N}-rep${k}" \
    -e SHAPES="${SHAPES:-1024:128 4096:256 8192:1024}" -e CONCURRENCIES="${CONCURRENCIES:-1 8 16}" &
done
wait
