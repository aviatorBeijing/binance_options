#!/bin/bash

SPAN=$1 #1d,1h, 15m etc.
ROLLING_WINDOW=120
$PYTHON spot_trading/actives.py  --wd $ROLLING_WINDOW --syms btc,eth,doge,ace,pendle,sei,sol --dt $SPAN --prev
