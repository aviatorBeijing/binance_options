#!/bin/bash

SERVER=3.114.152.67
# Go to server to run: binance_options/src/p.sh first

scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/doge-usdt_5m.csv ./ && mv doge-usdt_5m.csv ~/tmp
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/doge-usdt_1h.csv ./ && mv doge-usdt_1h.csv ~/tmp

scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/binance_*.csv ./
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/binance_fee_gain.dat ./
mv binance_*.csv ~/tmp
python spot_trading/portfolio.py --check_cached
python spot_trading/grid.py --ric doge/usdt --test --ref_spot=0.153 --start_ts 2024-04-13T22:30:00.000Z
