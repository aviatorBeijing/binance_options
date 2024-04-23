#!/bin/bash

$PYTHON spot_trading/bs_meta.py --centered_pair  --centered_pair_dist=25 --qty $1 --ric PENDLE/USDT
$PYTHON spot_trading/portfolio.py --ric PENDLE-USDT
