#!/bin/bash
# Benchmark script for the example scale kernel
# Usage: bash kernel_bench.sh [--quick]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QUICK=0
for arg in "$@"; do
  case $arg in
    --quick) QUICK=1 ;;
  esac
done

echo "=== Building kernel ==="
cd "$SCRIPT_DIR"
pip install -e . --quiet

echo "=== Running correctness tests ==="
python kernel_test.py

echo "=== Running benchmarks ==="
if [ "$QUICK" -eq 1 ]; then
  python benchmark.py --sizes 1024 4096 --dtypes float16 --warmup 10 --iters 50
else
  python benchmark.py
fi

echo "=== Done ==="
