#!/bin/bash

$PYTHON spot_trading/bs_meta.py --csell --ric=$1 --price $2 --qty $3
