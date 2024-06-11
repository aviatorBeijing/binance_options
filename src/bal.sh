#!/bin/bash

$PYTHON bbroker/check_balances.py

$PYTHON spot_trading/portfolio.py --hedging
