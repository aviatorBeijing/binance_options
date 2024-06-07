#!/bin/bash

arr=()
syms=()
datafiles=()
for s in BTC ETH BNB SOL XRP ADA AVAX LINK DOT TRX LTC FTM;do
	arr+=( "${s}/USDT" )
	syms+=( ${s,,} )
	datafiles+=( ${s,,}-usdt_1d.csv )
done

function join_by { local IFS="$1"; shift; echo "$*"; }

$PYTHON spot_trading/market_data.py --rics `join_by , ${arr[@]}` --span 1d

foodir=`pwd`
cd $USER_HOME/tmp
tar cvfz reversal_data.tar.gz `join_by " " ${datafiles[@]}` 
cd $foodir

$PYTHON signals/reversal_from_volume_hikes.py  --syms `join_by ,${syms[@]}` --volt 68
