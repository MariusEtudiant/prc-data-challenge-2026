#!/bin/bash
cd "$(dirname "$0")/.."
for A in LIRF LFPG LTFM EGLL EHAM EDDF LEBL LEMD LSZH EDDM; do
  echo "=== $A ==="
  PYTHONIOENCODING=utf-8 python -u scripts/exp_m23_optuna.py "$A" 30 2>&1 | grep -a -E "fit=|baseline tune|best tune|best params"
done
echo "RETUNE TERMINE"
