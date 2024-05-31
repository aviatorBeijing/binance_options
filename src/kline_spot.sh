#!/bin/bash

for s in DOGE/USDT PENDLE/USDT ACE/USDT SOL/USDT SEI/USDT;do
	for d in 1h 1d;do
		$PYTHON spot_trading/market_data.py --ric $s --span $d
	done
done

