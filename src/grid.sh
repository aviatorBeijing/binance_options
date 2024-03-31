#!/bin/bash

$PYTHON strategy/grid_trading.py --ric DOGE/USDT --nominal=10000 --stop_loss=-0.2 
$PYTHON strategy/grid_trading.py --ric SOL/USDT --nominal=1 --stop_loss=-0.2
$PYTHON strategy/grid_trading.py --ric ETH/USDT --nominal=1 --stop_loss=-0.2
$PYTHON strategy/grid_trading.py --ric BTC/USDT --nominal=1 --stop_loss=-0.2 
