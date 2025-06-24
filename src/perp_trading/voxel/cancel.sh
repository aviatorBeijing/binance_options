#!/bin/bash

RIC=VOXEL/USDT
$PYTHON perp_trading/perp_meta.py --cancel $1 --ric $RIC

