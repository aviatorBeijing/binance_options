#!/bin/bash

QTY=$1
RIC=SOL/USDT

$PYTHON perp_trading/perp_meta.py --ric $RIC --selldown $QTY
$PYTHON perp_trading/perp_meta.py --ric $RIC --check