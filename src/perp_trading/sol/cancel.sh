#!/bin/bash

RIC=SOL/USDT
$PYTHON perp_trading/perp_meta.py --cancel $1 --ric $RIC

