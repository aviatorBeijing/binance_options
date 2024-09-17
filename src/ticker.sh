#!/bin/bash
$PYTHON --version
RICS=$1

cd $BINANCE_OPTIONS_DIR
$PYTHON ws_bcontract.py --channel=ticker --ric $RICS
