#!/bin/bash

# Default values
SZ=0.01
IV=50
EXEC=false

# Parse named flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --date)
      DATE="$2"
      shift 2
      ;;
    --strike-call)
      STRIKE_CALL="$2"
      shift 2
      ;;
    --strike-put)
      STRIKE_PUT="$2"
      shift 2
      ;;
    --exec)
      EXEC=true
      shift 1
      ;;
    *)
      echo "Error: Unknown option $1"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$DATE" || -z "$STRIKE_PUT" || -z "$STRIKE_CALL" ]]; then
  echo "Error: --date and --strike are required."
  echo "Usage: $0 --date <DATE> --strike-call <STRIKE_CALL> --strike-put <STRIKE_PUT> [--exec]"
  exit 1
fi

# Build common arguments
ARGS=(
  "bbroker/straddle_calc.py"
  "--call" "BTC-${DATE}-${STRIKE_CALL}-C"
  "--put" "BTC-${DATE}-${STRIKE_PUT}-P"
  "--size" "$SZ"
  "--action" "buy"
  "--iv" "$IV"
)

# Append execute flag if requested
if [ "$EXEC" = true ]; then
  ARGS+=("--execute")
fi

# Execute script
$PYTHON "${ARGS[@]}"
