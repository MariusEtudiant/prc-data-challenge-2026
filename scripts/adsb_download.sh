#!/bin/bash
# One worker over an explicit list of days, so several can run at once on disjoint
# slices. The pull is network-bound (3.6 GB in 12 min is 5 MB/s, far under what gzip
# decompression sustains), so workers add throughput instead of competing for CPU.
cd "$(dirname "$0")/.."
for DAY in "$@"; do
  [ -f "data/adsb/ground_$DAY.parquet" ] && continue
  PYTHONIOENCODING=utf-8 python -u scripts/adsb_extract.py "$DAY" 2>&1 | grep --line-buffered -vi warning
done
echo "WORKER TERMINE: $1 .. ${@: -1}"
