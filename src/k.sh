#!/bin/bash

RIC=DOGE/USDT
$PYTHON spot_trading/kline_stats.py --ric $RIC --span 5m
