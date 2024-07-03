#!/bin/bash

SYMS=btc,doge,sol,xrp,eth,ada,link,bnb,atom,shib,ltc,ada,uni
MERC=gld
ASHARE=000729.SZ,000001.SZ,601186.SS,200550.SZ,600502.SS

# Signals will be stored in db="emmited_signals", table="signals".

#./cf.sh $SYMS	# Simulate Climb-Fall   algo

./vh.sh $SYMS	# Simulate Volume-Hikes algo
./vh.sh $ASHARE
./vh.sh `cat $USER_HOME/tmp/_rics_open_in_portfolio.csv`
./vh.sh $MERC

./ext.sh	# Collect mixed/rsi/sentiment signals (no simulation is done, ran separately to simulate.)
