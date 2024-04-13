#!/bin/bash

SERVER=3.114.152.67
# Go to server to run: binance_options/src/p.sh first
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/binance_*.csv ./
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/binance_fee_gain.dat ./
mv binance_*.csv ~/tmp
#python spot_trading/portfolio.py --check_cached
python spot_trading/grid.py --ric doge/usdt --test
