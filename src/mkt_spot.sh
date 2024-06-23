#!/bin/bash

RIC=DOGE/USDT

$PYTHON spot_trading/market_data.py --ric $RIC --span 1h
$PYTHON spot_trading/market_data.py --ric $RIC --span 1d

$PYTHON perp_trading/marketdata.py  --ric $RIC --span 1h
$PYTHON perp_trading/marketdata.py  --ric $RIC --span 1d

RIC=BTC/USDT
$PYTHON spot_trading/market_data.py --ric $RIC --span 1h
$PYTHON spot_trading/market_data.py --ric $RIC --span 1d

$PYTHON perp_trading/marketdata.py  --ric $RIC --span 1h
$PYTHON perp_trading/marketdata.py  --ric $RIC --span 1d

RIC=MATIC/USDT
$PYTHON spot_trading/market_data.py --ric $RIC --span 1h
$PYTHON spot_trading/market_data.py --ric $RIC --span 1d

$PYTHON perp_trading/marketdata.py  --ric $RIC --span 1h
$PYTHON perp_trading/marketdata.py  --ric $RIC --span 1d

./reversal.sh 60
./kline_spot.sh
./gain_fee.sh
