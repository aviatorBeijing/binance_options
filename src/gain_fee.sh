#!/bin/bash

function _fee(){
   cd spot_trading/$1/
   ./p
   cd ../..
}

function _dat(){
	$PYTHON -c 'import pandas as pd;import os;fp=open(os.getenv("USER_HOME","")+"/tmp/binance_fee_gain_'$1'-usdt.dat","r");lines=fp.readlines(); rec = list(map(lambda ln:ln.strip().split(":"), lines));df=pd.DataFrame(rec).set_index(0).transpose();from tabulate import tabulate;print( tabulate(df,headers="keys") )' 
}

echo "reading:"
for s in doge sol pendle ace sei btc matic
do
        _fee $s
done 

echo "check:"
for s in doge sol pendle ace sei btc matic
do
	_dat $s
done 

