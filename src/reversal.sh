#!/bin/bash

for s in BTC ETH BNB SOL XRP ADA AVAX LINK DOT TRX;do
       echo
       #$PYTHON spot_trading/market_data.py --ric $s/USDT --span 1d	
done

$PYTHON signals/reversal_from_longterm_low.py --syms btc,eth,bnb,sol,xrp,ada,avax,link,dot,trx --volt $1
