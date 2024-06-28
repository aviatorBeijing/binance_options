#!/bin/bash

$PYTHON spot_trading/portfolio.py --hedging

#$PYTHON spot_trading/portfolio.py --hedging --var_cryptos sei,ace,pendle

$PYTHON spot_trading/portfolio.py --hedging --var_cryptos sol,btc,doge,bnb,eth,usdt
