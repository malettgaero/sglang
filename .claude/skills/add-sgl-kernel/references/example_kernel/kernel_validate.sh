#!/bin/bash
# Validation script for the scale kernel
# Runs correctness checks and shape validation against PyTorch reference

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================="
echo " Scale Kernel Validation"
echo "======================================="

# Default options
RUN_CORRECTNESS=1
RUN_SHAPES=1
RUN_DTYPES=1
VERBOSE=0

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --correctness-only   Run only correctness checks"
    echo "  --shapes-only        Run only shape mismatch checks"
    echo "  --dtypes-only        Run only dtype checks"
    echo "  --verbose            Enable verbose output"
    echo "  -h, --help           Show this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --correctness-only)
            RUN_SHAPES=0
            RUN_DTYPES=0
            shift
            ;;
        --shapes-only)
            RUN_CORRECTNESS=0
            RUN_DTYPES=0
            shift
            ;;
        --dtypes-only)
            RUN_CORRECTNESS=0
            RUN_SHAPES=0
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

cd "$SCRIPT_DIR"

VERBOSE_FLAG=""
if [[ $VERBOSE -eq 1 ]]; then
    VERBOSE_FLAG="--verbose"
fi

PASSED=0
FAILED=0

run_check() {
    local name="$1"
    local cmd="$2"
    echo -n "  Running $name ... "
    if eval "$cmd" > /tmp/validate_out.txt 2>&1; then
        echo "PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "FAILED"
        FAILED=$((FAILED + 1))
        echo "  --- output ---"
        cat /tmp/validate_out.txt | sed 's/^/  | /'
        echo "  -------------"
    fi
}

if [[ $RUN_CORRECTNESS -eq 1 ]]; then
    echo ""
    echo "[1] Correctness validation"
    run_check "float32 correctness" \
        "python kernel_validate.py --check correctness --dtype float32 $VERBOSE_FLAG"
    run_check "float16 correctness" \
        "python kernel_validate.py --check correctness --dtype float16 $VERBOSE_FLAG"
    run_check "bfloat16 correctness" \
        "python kernel_validate.py --check correctness --dtype bfloat16 $VERBOSE_FLAG"
fi

if [[ $RUN_SHAPES -eq 1 ]]; then
    echo ""
    echo "[2] Shape validation"
    run_check "shape mismatch detection" \
        "python kernel_validate.py --check shapes $VERBOSE_FLAG"
fi

if [[ $RUN_DTYPES -eq 1 ]]; then
    echo ""
    echo "[3] Dtype validation"
    run_check "dtype handling" \
        "python kernel_validate.py --check dtypes $VERBOSE_FLAG"
fi

echo ""
echo "======================================="
echo " Results: $PASSED passed, $FAILED failed"
echo "======================================="

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
