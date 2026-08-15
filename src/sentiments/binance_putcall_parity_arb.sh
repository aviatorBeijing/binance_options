#!/bin/bash

export PYTHONPATH=/home/ubuntu/src/binance_options/src:$PYTHONPATH
$PYTHON binance_putcall_parity_arb.py --check_parity
