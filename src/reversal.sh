#!/bin/bash

arr=()
syms=()
for s in BTC ETH BNB SOL XRP ADA AVAX LINK DOT TRX;do
	arr+=( "${s}/USDT" )
	syms+=( ${s,,} )
       #$PYTHON spot_trading/market_data.py --ric $s/USDT --span 1d	
done

function join_by { local IFS="$1"; shift; echo "$*"; }

$PYTHON spot_trading/market_data.py --rics `join_by , ${arr[@]}` --span 1d

$PYTHON signals/reversal_from_longterm_low.py --syms `join_by ,${syms[@]}` --volt 68
