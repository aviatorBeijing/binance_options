#!/bin/bash

SYMS=btc,doge,sol,xrp,eth

# Signals will be stored in db="emmited_signals", table="signals".

#./cf.sh $SYMS	# Simulate Climb-Fall   algo
./vh.sh $SYMS	# Simulate Volume-Hikes algo
./ext.sh	# Collect mixed/rsi/sentiment signals (no simulation is done, ran separately to simulate.)
