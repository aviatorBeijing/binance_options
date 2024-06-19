#!/bin/bash

WTS=$2
CRYPTOS=$1
#scyb,igib
#gld
#btc,doge

$PYTHON signals/cvar.py --cryptos $CRYPTOS --weights $WTS
