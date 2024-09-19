#!/bin/bash

cd $BINANCE_OPTIONS_DIR

CTR=$1 # BTC-240416-71000-C
COST=$2 # 100
$PYTHON brisk/pricing.py --contract=$CTR --user_cost $COST
