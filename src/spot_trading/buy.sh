#!/bin/bash

$PYTHON spot_trading/bs_meta.py --cbuy --ric=$1 --price $2 --qty $3
