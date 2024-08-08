#!/bin/bash

for s in sol pendle ace sei avax trx btc matic xrp doge bnb;do
    $PYTHON spot_trading/portfolio.py --check_cached --ric $s-usdt &
done
