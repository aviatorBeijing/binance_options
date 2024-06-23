#!/bin/bash

# !!!! The portfolio graph is refreshed after a (any) crypto check.

python spot_trading/portfolio.py --check_assets
python spot_trading/portfolio.py --hedging --var_cryptos btc,eth,sol,doge,usdt,matic,xrp
