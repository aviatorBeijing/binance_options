#!/bin/bash

RIC=DOGE/USDT

$PYTHON spot_trading/market_data.py --ric $RIC --span 5m
$PYTHON spot_trading/market_data.py --ric $RIC --span 1h
