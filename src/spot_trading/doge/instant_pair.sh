#!/bin/bash

$PYTHON spot_trading/bs_meta.py --centered_pair  --qty $1 --ric DOGE/USDT --centered_pair_dist=25
$PYTHON spot_trading/portfolio.py
