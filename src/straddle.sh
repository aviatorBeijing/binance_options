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
    --strike)
      STRIKE="$2"
      shift 2
      ;;
    --exec)
      EXEC=true
      shift 1
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build common arguments
ARGS=(
  "bbroker/straddle_calc.py"
  "--call" "BTC-${DATE}-${STRIKE}-C"
  "--put" "BTC-${DATE}-${STRIKE}-P"
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
