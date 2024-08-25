#!/bin/bash

SYMS=btc,doge,sol,xrp,eth,ada,link,bnb,atom,shib,ltc,ada,uni
MERC=gld
ASHARE=000729.SZ,000001.SZ,601186.SS,200550.SZ,600502.SS

PWD=`pwd`

cd $FINAPI/api/strategies/scripts

# Signals will be stored in db="emmited_signals", table="signals".

#./cf.sh $SYMS	# Simulate Climb-Fall   algo

./fin_vh.sh 4151.T,4502.T,4568.T,4578.T
./fin_vh.sh $SYMS	# Simulate Volume-Hikes algo
./fin_vh.sh msft,goog,nvda,cdns,amzn,nvo,cere,lbrdk,armk,j,bkng,tsla,^gspc,^fvx,gld,'bz=f'
#./fin_vh.sh `cat $USER_HOME/tmp/_rics_open_in_portfolio.csv`
./fin_vh.sh $MERC


cd $FINAPI/api/scripts
$PYTHON ext_strategies.py
#./ext.sh	# Collect mixed/rsi/sentiment signals (no simulation is done, ran separately to simulate.)
