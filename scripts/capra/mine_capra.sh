#!/usr/bin/env bash
# mine_capra.sh
# Offline CAPRA supervision mining.
#
# Usage:
#   bash scripts/capra/mine_capra.sh [CHECKPOINT] [DATASET] [CACHE_ROOT]
#
# Defaults point at placeholder paths; override via positional args or
# by editing the variables below before uploading to the server.

set -euo pipefail

CHECKPOINT="${1:-tmp/models/openvla-oft-libero}"
DATASET="${2:-libero_spatial}"
CACHE_ROOT="${3:-tmp/capra_cache}"

echo "[mine_capra] checkpoint : $CHECKPOINT"
echo "[mine_capra] dataset    : $DATASET"
echo "[mine_capra] cache_root : $CACHE_ROOT"

python -m experiments.robot.capra.run_capra_mining \
    --pretrained_checkpoint "$CHECKPOINT" \
    --dataset_name          "$DATASET" \
    --cache_root            "$CACHE_ROOT" \
    --num_mining_episodes   50 \
    --K                     8 \
    --H_s                   5 \
    --W                     10 \
    --seed                  7
