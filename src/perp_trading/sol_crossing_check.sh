#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$USER_HOME/src/binance_options/src

RIC=SOL/USDT
$PYTHON perp_trading/marketdata.py  --ric $RIC --span 15m
$PYTHON perp_trading/sol_crossing_check.py 
