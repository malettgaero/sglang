#!/bin/bash
# Profile the scale kernel using Nsight Systems or Nsight Compute.
# Usage:
#   ./kernel_profile.sh [nsys|ncu] [options]
#
# Examples:
#   ./kernel_profile.sh nsys
#   ./kernel_profile.sh ncu --metric l1tex__t_bytes

set -euo pipefail

MODE=${1:-nsys}
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_SCRIPT="${SCRIPT_DIR}/kernel_profile.py"

if [[ ! -f "${PROFILE_SCRIPT}" ]]; then
  echo "ERROR: kernel_profile.py not found at ${PROFILE_SCRIPT}" >&2
  exit 1
fi

case "${MODE}" in
  nsys)
    OUTPUT="profile_scale_nsys"
    echo "Running Nsight Systems profile -> ${OUTPUT}.nsys-rep"
    nsys profile \
      --output="${OUTPUT}" \
      --force-overwrite=true \
      --trace=cuda,nvtx \
      --stats=true \
      python "${PROFILE_SCRIPT}" --mode nsight "$@"
    echo "Done. Open with: nsys-ui ${OUTPUT}.nsys-rep"
    ;;
  ncu)
    OUTPUT="profile_scale_ncu"
    echo "Running Nsight Compute profile -> ${OUTPUT}.ncu-rep"
    ncu \
      --output="${OUTPUT}" \
      --force-overwrite \
      --set full \
      "$@" \
      python "${PROFILE_SCRIPT}" --mode nsight
    echo "Done. Open with: ncu-ui ${OUTPUT}.ncu-rep"
    ;;
  *)
    echo "Unknown mode '${MODE}'. Use 'nsys' or 'ncu'." >&2
    exit 1
    ;;
esac
