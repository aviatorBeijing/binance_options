#!/bin/bash

QTY=$1
RIC=XMR/USDT
$PYTHON perp_trading/perp_meta.py --ric $RIC --centered_pair --centered_pair_dist=25 --qty=$QTY
$PYTHON perp_trading/perp_meta.py --ric $RIC --check
