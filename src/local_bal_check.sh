#!/bin/bash

# !!!! The portfolio graph is refreshed after a (any) crypto check.

python spot_trading/portfolio.py --check_cached --ric sol-usdt

python spot_trading/portfolio.py --hedging --var_cryptos btc,eth,sol,doge,usdt,matic
