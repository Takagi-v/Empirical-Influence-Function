#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
IB-FT is a training-objective baseline, not a data-rewrite baseline.

The reusable IB-FT loss module is implemented in:
  src/baseline/ibft.py

To run a real IB-FT experiment, wire VariationalBottleneck +
compute_ibft_loss into the trainer used by your benchmark owner.  I am leaving
this entry as an explicit guard instead of silently launching ordinary CE
training and pretending it is IB-FT.
EOF
exit 2

