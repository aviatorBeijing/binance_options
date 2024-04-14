#!/bin/bash

RIC=DOGE/USDT
QTY=50

PCE_BUY=$1
PCE_SELL=$2

$PYTHON spot_trading/bs_meta.py --cbuy --ric=$RIC --price $PCE_BUY --qty $QTY

$PYTHON spot_trading/bs_meta.py --csell --ric=$RIC --price $PCE_SELL --qty $QTY
