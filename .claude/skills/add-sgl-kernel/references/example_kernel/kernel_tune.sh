#!/bin/bash
# Tuning script for the scale kernel
# Runs kernel_tune.py with various configurations and logs results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/tune_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/tune_${TIMESTAMP}.json"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Scale Kernel Tuning"
echo "Timestamp: ${TIMESTAMP}"
echo "Output: ${OUTPUT_FILE}"
echo "=========================================="

# Default sizes to tune over
SIZES=(1024 4096 16384 65536 262144 1048576)

# Parse optional args
WHILE_SIZES=()
for arg in "$@"; do
    case $arg in
        --sizes=*)
            IFS=',' read -ra WHILE_SIZES <<< "${arg#*=}"
            ;;
        --output=*)
            OUTPUT_FILE="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

if [ ${#WHILE_SIZES[@]} -gt 0 ]; then
    SIZES=("${WHILE_SIZES[@]}")
fi

SIZES_ARG=$(IFS=','; echo "${SIZES[*]}")

echo "Tuning over sizes: ${SIZES_ARG}"
echo ""

python "${SCRIPT_DIR}/kernel_tune.py" \
    --sizes "${SIZES_ARG}" \
    --output "${OUTPUT_FILE}"

echo ""
echo "Tuning complete. Results saved to: ${OUTPUT_FILE}"
