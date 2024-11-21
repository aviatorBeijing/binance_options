#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$USER_HOME/src/binance_options/src

RIC=DOGE/USDT
$PYTHON perp_trading/marketdata.py  --ric $RIC --span 15m
$PYTHON perp_trading/price_crossing_check.py --sym doge
