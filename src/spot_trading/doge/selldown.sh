#!/bin/bash

QTY=$1
$PYTHON spot_trading/bs_meta.py --ric DOGE-USDT --selldown $QTY
