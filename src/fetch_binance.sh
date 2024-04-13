#!/bin/bash

scp -i ~/.ssh/junma-japan.pem ubuntu@3.114.152.67:/home/ubuntu/tmp/binance_*.csv ./
scp -i ~/.ssh/junma-japan.pem ubuntu@3.114.152.67:/home/ubuntu/tmp/binance_fee_gain.dat ./
