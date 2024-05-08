#!/bin/bash

QTY=$1

$PYTHON perp_trading/perp_meta.py --ric DOGE/USDT --centered_pair --centered_pair_dist=25 --qty=$QTY
