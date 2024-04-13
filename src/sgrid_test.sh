#!/bin/bash

START_TS=2024-04-12T18:40:00.000Z
GAP=52

export GRID_DEBUG=
python spot_trading/grid.py --ric doge/usdt --start_ts $START_TS --test --span=5m --uniform_grid_gap $GAP
