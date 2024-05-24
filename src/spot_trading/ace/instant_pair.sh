#!/bin/bash

$PYTHON spot_trading/bs_meta.py --centered_pair  --centered_pair_dist=35 --qty $1 --ric ACE/USDT
$PYTHON spot_trading/portfolio.py --ric ACE-USDT
