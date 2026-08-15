#!/bin/bash

$PYTHON binance_putcall_parity_arb.py --refresh_oi --underlying $1

$PYTHON binance_putcall_parity_arb.py  --underlying $1 --check_parity
