#!/bin/bash

QTY=$1
RIC=AERGO/USDT

$PYTHON perp_trading/perp_meta.py --ric $RIC --selldown $QTY
$PYTHON perp_trading/perp_meta.py --ric $RIC --check
