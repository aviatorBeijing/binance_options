#!/bin/bash

# Acquire websocket data, cached in local files
export BINANCE_DEBUG=1
$PYTHON strategy/scan_straddle.py --contracts `cat ~/tmp/_contracts_test.csv` --data
