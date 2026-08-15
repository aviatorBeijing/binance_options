#!/bin/bash

$PYTHON binance_putcall_parity_arb.py --refresh_oi --underlying ETH

$PYTHON binance_putcall_parity_arb.py  --underlying ETH --check_price_ranges
