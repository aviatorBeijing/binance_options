#!/bin/bash
# Ref: https://www.coinglass.com/ArbitrageList
echo "Check oppr. on: "
echo "         https://www.coinglass.com/ArbitrageList"
echo "Example: ./arb.sh mtl"
echo 

$PYTHON arb/perp_spot_arg.py --sym $1 
