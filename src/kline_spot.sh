#!/bin/bash

for s in DOGE PENDLE ACE SOL SEI;do
	for d in 1h 1d;do
		$PYTHON spot_trading/market_data.py --ric $s/USDT --span $d
	done
done

