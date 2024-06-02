#!/bin/bash
SERVER=3.114.152.67
for s in btc eth bnb sol xrp ada avax link dot trx;do
   scp -i ~/.ssh/junma-japan.pem ubuntu@$SERVER:/home/ubuntu/tmp/${s}-usdt_1d.csv ./ && mv ${s}-usdt_1d.csv ~/tmp
done

