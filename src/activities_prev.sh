#!/bin/bash

SPAN=$1 #1d,1h, 15m etc.
ROLLING_WINDOW=120
$PYTHON spot_trading/actives.py  --wd $ROLLING_WINDOW --syms btc,eth,doge,ace,pendle,sei,sol,bnb,xrp,ada,avax,link,dot,trx --dt $SPAN --prev
#BTC ETH BNB SOL XRP ADA AVAX LINK DOT TRX;
