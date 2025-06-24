#!/bin/bash

# /home/ubuntu/src/binance_options/src/ticker.sh BTC-250725-104000-C,BTC-250725-104000-P

CONTRACTS=BTC-250725-104000-C,BTC-250725-104000-P

$PYTHON ivsurf/on_spot.py --contracts $CONTRACTS \
			--cap_call 1000 \
			--cap_put 1000 \
			--alloc 1,1,1,4,8
