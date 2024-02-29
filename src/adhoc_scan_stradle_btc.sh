#!/bin/bash

# Run adhoc_scan_stradle_btc_data.sh first in a separate terminal (or screen)
# to collect market data in background, then
# run this script.

$PYTHON strategy/scan_straddle.py --contracts `cat ~/tmp/_contracts_test.csv`
