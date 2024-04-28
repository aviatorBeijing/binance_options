#!/bin/bash

$PYTHON spot_trading/bs_meta.py --csell --ric=PENDLE/USDT --price $1 --qty $2
