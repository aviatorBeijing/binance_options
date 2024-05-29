#!/bin/bash

for s in DOGE/USDT PENDLE/USDT ACE/USDT SOL/USDT SEI/USDT;do
	$PYTHON spot_trading/market_data.py --ric $s --span 1h
done

