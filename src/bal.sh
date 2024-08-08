#!/bin/bash

$PYTHON bbroker/check_balances.py

$PYTHON spot_trading/portfolio.py --hedging

$PYTHON spot_trading/portfolio.py  --check_assets
