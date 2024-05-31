#!/bin/bash

SERVER=3.114.152.67
# Go to server to run: binance_options/src/p.sh first

scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/bal.csv ./ && mv bal.csv ~/tmp

for s in sol-usdt pendle-usdt sei-usdt ace-usdt;do
	scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/${s}_1h.csv ./ && mv ${s}_1h.csv ~/tmp
done

scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/doge-usdt_5m.csv ./ && mv doge-usdt_5m.csv ~/tmp
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/doge-usdt_1h.csv ./ && mv doge-usdt_1h.csv ~/tmp

scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/doge-usdt_1d.csv ./ && mv doge-usdt_1d.csv ~/tmp
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/sei-usdt_1d.csv ./ && mv sei-usdt_1d.csv ~/tmp
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/sol-usdt_1d.csv ./ && mv sol-usdt_1d.csv ~/tmp

scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/btc-usdt_1h.csv ./ && mv btc-usdt_1h.csv ~/tmp
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/btc-usdt_1d.csv ./ && mv btc-usdt_1d.csv ~/tmp

scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/perp_dogeusdt_5m.csv ./ && mv perp_dogeusdt_5m.csv ~/tmp
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/perp_dogeusdt_1h.csv ./ && mv perp_dogeusdt_1h.csv ~/tmp

scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/binance_*.csv ./
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/binance_fee_gain.dat ./
mv binance_*.csv ~/tmp
python spot_trading/portfolio.py --check_cached
python spot_trading/grid.py --ric doge/usdt --test --ref_spot=0.13908 --start_ts 2024-04-13T22:30:00.000Z

for s in sol-usdt pendle-usdt ace-usdt sei-usdt;do
    python spot_trading/portfolio.py --check_cached --ric $s
done
