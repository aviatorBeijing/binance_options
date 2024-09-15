#!/bin/bash

PCS=$2
CONTRACTS=$1

export PYTHONPATH=/home/ubuntu/src/binance_options/src:$PYTHONPATH
cd /home/ubuntu/src/binance_options/src/strategy/options_pricing
$PYTHON options_price_projection_from_spot_price.py --contracts $CONTRACTS --projected_spot_prices $PCS
