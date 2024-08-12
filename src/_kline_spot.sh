#!/bin/bash

SYM=$1
SPAN=$2
$PYTHON spot_trading/market_data.py --ric $SYM/USDT --span $SPAN

