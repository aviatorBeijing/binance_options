#!/bin/bash
source _settings.sh
cd $WDIR

$PYTHON spot_trading/bs_meta.py --ric $SYM/USDT --sellbest --qty=$1
