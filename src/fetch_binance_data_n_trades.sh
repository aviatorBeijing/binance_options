#!/bin/bash

SERVER=3.114.152.67
# Go to server to run: binance_options/src/p.sh first

scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/bal.csv ./ && mv bal.csv ~/tmp

scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/doge-usdt_5m.csv ./ && mv doge-usdt_5m.csv ~/tmp

#for s in doge sei ace pendle sol avax trx btc matic;do
#  scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/$s-usdt_1d.csv ./ && mv $s-usdt_1d.csv ~/tmp
#  #scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/$s-usdt_1h.csv ./ && mv $s-usdt_1h.csv ~/tmp
#done

scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/perp_dogeusdt_5m.csv ./ && mv perp_dogeusdt_5m.csv ~/tmp
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/perp_dogeusdt_1h.csv ./ && mv perp_dogeusdt_1h.csv ~/tmp

CPWD=`pwd`
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/binance_kline.tar.gz ./;mv binance_kline.tar.gz ~/tmp;cd ~/tmp;tar xvfz binance_kline.tar.gz;cd $CPWD
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/binance_ot.tar.gz ./;mv binance_ot.tar.gz ~/tmp;cd ~/tmp;tar xvfz binance_ot.tar.gz;cd $CPWD
scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/binance_fee_gain.dat ./

cd $BINANCE_OPTIONS_DIR
$PYTHON spot_trading/portfolio.py  --check_assets
./win_losses.sh

#python spot_trading/grid.py --ric doge/usdt --test --ref_spot=0.13908 --start_ts 2024-04-13T22:30:00.000Z
