#!/usr/bin/env bash
# smoke_capra.sh -- Quick smoke test: verify pure-Python CAPRA logic.
# Does NOT require a GPU, a model checkpoint, or a LIBERO install.
#
# Covers:
#   1.  CAPRAConfig + FinetuneCAPRAConfig
#   2.  Equivalence filter
#   3.  Safety target distribution
#   4.  CAPRA KL loss (CPU tensors)
#   5.  Training branches: baseline / CAPRA / anchor-only
#   6.  SPIR / EAR metrics
#   7.  EpisodeMetrics + AggregateMetrics
#   8.  Report writers (JSON, CSV, Markdown)
#   9.  Precursor weight
#   10. Procedural splits: all 4 templates with mock env
#   11. CAPRAEnvAdapter no-sim path
#
# Usage:
#   bash scripts/capra/smoke_capra.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[smoke_capra] repo: $REPO_ROOT"
echo "[smoke_capra] Running pure-Python CAPRA logic checks..."

python scripts/capra/_smoke_logic.py

echo "[smoke_capra] All checks passed."
