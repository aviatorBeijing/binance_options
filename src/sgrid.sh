#!/bin/bash

RIC=doge/usdt,btc/usdt,bnb/usdt,sol/usdt
SPAN=5m
GRID_GAP_BPS=100
START_TS=2024-04-13T23:30:00.000Z

$PYTHON spot_trading/grid.py --rics $RIC --span=$SPAN --uniform_grid_gap $GRID_GAP_BPS --start_ts=$START_TS
