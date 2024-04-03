#!/bin/bash
SPANS=$1
$PYTHON strategy/grid_trading.py --test --ric DOGE/USDT --nominal=10000 --stop_loss=-0.2  --spans=$SPANS
$PYTHON strategy/grid_trading.py --test --ric SOL/USDT --nominal=100 --stop_loss=-0.2 --spans=$SPANS
$PYTHON strategy/grid_trading.py --test --ric ETH/USDT --nominal=1 --stop_loss=-0.2 --spans=$SPANS
$PYTHON strategy/grid_trading.py --test --ric LTC/USDT --nominal=100 --stop_loss=-0.2 --spans=$SPANS
$PYTHON strategy/grid_trading.py --test --ric BTC/USDT --nominal=1 --stop_loss=-0.2  --spans=$SPANS
